#!/usr/bin/env python3
"""
DCGAN con Discriminador Preentrenado - Script de Entrenamiento Mejorado
=======================================================================

Este script implementa una DCGAN con opción de usar discriminadores preentrenados
de torchvision para generar imágenes de alta calidad en datasets como FashionMNIST.

Mejoras incluidas:
- Logging avanzado con métricas detalladas
- Visualización en tiempo real del progreso
- Soporte para múltiples datasets
- Regularización mejorada
- Checkpointing robusto
- Evaluación automática de calidad

Autor: Sistema de IA
Fecha: 2025
"""

import argparse
import os
import math
import random
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils, models

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm no disponible. Instala con: pip install tqdm")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib/seaborn no disponible. Algunas visualizaciones no estarán disponibles")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy no disponible. Algunas funciones estadísticas no estarán disponibles")

def set_seed(seed: int = 42):
    """Establecer semilla para reproducibilidad completa"""
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir: Path) -> None:
    """Configurar logging en archivo"""
    import logging
    
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def weights_init_dcgan(m):
    """Inicialización de pesos DCGAN estándar"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init_xavier(m):
    """Inicialización Xavier/Glorot para redes profundas"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def save_checkpoint(state: Dict[str, Any], outdir: Path, filename: str) -> None:
    """Guardar checkpoint con validación"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = outdir / filename
    torch.save(state, checkpoint_path)
    
    # Crear enlace simbólico al último checkpoint
    latest_path = outdir / "latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    try:
        latest_path.symlink_to(filename)
    except OSError:
        # En Windows, copiar el archivo si no se pueden crear symlinks
        import shutil
        shutil.copy2(checkpoint_path, latest_path)

def denorm(x: torch.Tensor) -> torch.Tensor:
    """Desnormalizar de [-1,1] a [0,1]"""
    return x.add(1).div(2).clamp(0, 1)

def add_noise(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """Agregar ruido gaussiano para regularización"""
    if noise_std > 0:
        return x + torch.randn_like(x) * noise_std
    return x

def count_parameters(model: nn.Module) -> int:
    """Contar parámetros entrenables del modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Generator(nn.Module):
    """
    Generador DCGAN mejorado con arquitectura optimizada
    """
    def __init__(self, z_dim=100, img_channels=3, feature_g=64, out_size=64):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.feature_g = feature_g
        
        # Arquitectura DCGAN estándar con mejoras
        self.net = nn.Sequential(
            # Input: z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 64x64
        )
        
    def forward(self, z):
        return self.net(z)
    
    def generate_samples(self, num_samples: int = 16, device: str = 'cpu') -> torch.Tensor:
        """Generar muestras aleatorias"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim, 1, 1, device=device)
            return self(z)

class DiscriminatorDCGAN(nn.Module):
    """
    Discriminador DCGAN clásico con dropout mejorado
    """
    def __init__(self, img_channels=3, feature_d=64, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.net = nn.Sequential(
            # 64x64
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 32x32
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 16x16
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 8x8
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 4x4
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
            # 1x1
        )
        
    def forward(self, x):
        output = self.net(x)
        return output.view(x.size(0), -1).squeeze(1)

class DiscriminatorPretrained(nn.Module):
    """
    Discriminador con backbone preentrenado y cabezal mejorado
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, dropout_rate=0.2):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        
        self.backbone, feat_dim = self._make_backbone(self.backbone_name, pretrained)
        
        # Cabezal clasificador mejorado con capas residuales
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim // 4, 1)
        )
        
    def _make_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Crear backbone con manejo mejorado de errores"""
        weights = None
        try:
            if pretrained:
                if name == "resnet18":
                    weights = models.ResNet18_Weights.DEFAULT
                elif name == "mobilenet_v2":
                    weights = models.MobileNet_V2_Weights.DEFAULT
                elif name == "efficientnet_b0":
                    weights = models.EfficientNet_B0_Weights.DEFAULT
                elif name == "vgg11_bn":
                    weights = models.VGG11_BN_Weights.DEFAULT
        except Exception as e:
            print(f"⚠️  No se pudieron cargar pesos pretrained para {name}: {e}")
            weights = None
            
        if name == "resnet18":
            m = models.resnet18(weights=weights)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, feat_dim
        elif name == "mobilenet_v2":
            m = models.mobilenet_v2(weights=weights)
            feat_dim = m.classifier[-1].in_features
            m.classifier = nn.Identity()
            return m, feat_dim
        elif name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=weights)
            feat_dim = m.classifier[-1].in_features
            m.classifier = nn.Identity()
            return m, feat_dim
        elif name == "vgg11_bn":
            m = models.vgg11_bn(weights=weights)
            feat_dim = 512  # VGG features antes del classificador
            m.classifier = nn.Identity()
            return m, feat_dim
        else:
            raise ValueError(f"Backbone no soportado: {name}")
            
    def forward(self, x):
        feat = self.backbone(x)
        if isinstance(feat, torch.Tensor) and feat.dim() > 2:
            feat = torch.flatten(feat, 1)
        return self.classifier(feat).squeeze(1)

def get_dataset(name="fashion_mnist", root="./data", image_size=64):
    """
    Obtener dataset con transformaciones optimizadas
    """
    name = name.lower()
    
    # Transformaciones específicas por dataset
    if name == "cifar10":
        # CIFAR-10 ya es en color
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        # Para datasets en escala de grises
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convertir a RGB
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    if name in ["fashion_mnist", "fashion", "fmnist"]:
        return datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    elif name == "mnist":
        return datasets.MNIST(root=root, train=True, download=True, transform=transform)
    elif name == "kmnist":
        return datasets.KMNIST(root=root, train=True, download=True, transform=transform)
    elif name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset no soportado: {name}. Use fashion_mnist|mnist|kmnist|cifar10.")

class TrainingMetrics:
    """Clase para manejar métricas de entrenamiento"""
    def __init__(self):
        self.losses_D = []
        self.losses_G = []
        self.scores_real = []
        self.scores_fake = []
        self.lr_G = []
        self.lr_D = []
        self.epoch_times = []
        self.gradient_norms_G = []
        self.gradient_norms_D = []
    
    def update(self, loss_D, loss_G, score_real=None, score_fake=None, 
               lr_G=None, lr_D=None, grad_norm_G=None, grad_norm_D=None):
        """Actualizar métricas"""
        self.losses_D.append(loss_D)
        self.losses_G.append(loss_G)
        if score_real is not None:
            self.scores_real.append(score_real)
        if score_fake is not None:
            self.scores_fake.append(score_fake)
        if lr_G is not None:
            self.lr_G.append(lr_G)
        if lr_D is not None:
            self.lr_D.append(lr_D)
        if grad_norm_G is not None:
            self.gradient_norms_G.append(grad_norm_G)
        if grad_norm_D is not None:
            self.gradient_norms_D.append(grad_norm_D)
    
    def add_epoch_time(self, time_seconds):
        """Agregar tiempo de época"""
        self.epoch_times.append(time_seconds)
    
    def save_to_json(self, filepath):
        """Guardar métricas en JSON"""
        data = {
            'losses_D': self.losses_D,
            'losses_G': self.losses_G,
            'scores_real': self.scores_real,
            'scores_fake': self.scores_fake,
            'lr_G': self.lr_G,
            'lr_D': self.lr_D,
            'epoch_times': self.epoch_times,
            'gradient_norms_G': self.gradient_norms_G,
            'gradient_norms_D': self.gradient_norms_D
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot_metrics(self, save_path=None):
        """Plotear métricas de entrenamiento"""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib no disponible para plotting")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Pérdidas
        axes[0, 0].plot(self.losses_D, label='Discriminador', alpha=0.7)
        axes[0, 0].plot(self.losses_G, label='Generador', alpha=0.7)
        axes[0, 0].set_title('Pérdidas')
        axes[0, 0].set_xlabel('Iteración')
        axes[0, 0].set_ylabel('Pérdida')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scores
        if self.scores_real and self.scores_fake:
            axes[0, 1].plot(self.scores_real, label='Real', alpha=0.7)
            axes[0, 1].plot(self.scores_fake, label='Falso', alpha=0.7)
            axes[0, 1].set_title('Scores del Discriminador')
            axes[0, 1].set_xlabel('Iteración')
            axes[0, 1].set_ylabel('Score promedio')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rates
        if self.lr_G and self.lr_D:
            epochs = range(len(self.lr_G))
            axes[0, 2].plot(epochs, self.lr_G, label='LR Generador', alpha=0.7)
            axes[0, 2].plot(epochs, self.lr_D, label='LR Discriminador', alpha=0.7)
            axes[0, 2].set_title('Learning Rates')
            axes[0, 2].set_xlabel('Época')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Tiempos por época
        if self.epoch_times:
            axes[1, 0].plot(self.epoch_times, marker='o', alpha=0.7)
            axes[1, 0].set_title('Tiempo por Época')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Tiempo (segundos)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Normas de gradientes
        if self.gradient_norms_G and self.gradient_norms_D:
            axes[1, 1].plot(self.gradient_norms_G, label='Generador', alpha=0.7)
            axes[1, 1].plot(self.gradient_norms_D, label='Discriminador', alpha=0.7)
            axes[1, 1].set_title('Normas de Gradientes')
            axes[1, 1].set_xlabel('Iteración')
            axes[1, 1].set_ylabel('Norma L2')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Distribución de pérdidas (histograma)
        axes[1, 2].hist(self.losses_D[-1000:], bins=30, alpha=0.7, label='D', density=True)
        axes[1, 2].hist(self.losses_G[-1000:], bins=30, alpha=0.7, label='G', density=True)
        axes[1, 2].set_title('Distribución de Pérdidas (últimas 1000)')
        axes[1, 2].set_xlabel('Pérdida')
        axes[1, 2].set_ylabel('Densidad')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def calculate_gradient_norm(model: nn.Module) -> float:
    """Calcular norma L2 de los gradientes"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def train(args):
    """
    Función principal de entrenamiento mejorada
    """
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"🚀 Usando dispositivo: {device}")
    
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Configurar directorios
    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    
    for dir_path in [samples_dir, ckpt_dir, logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Configurar logging
    logger = setup_logging(logs_dir)
    logger.info(f"Iniciando entrenamiento con configuración: {vars(args)}")
    
    # Reproducibilidad
    set_seed(args.seed)
    
    # Cargar dataset
    logger.info(f"Cargando dataset: {args.dataset}")
    train_ds = get_dataset(args.dataset, args.data_dir, args.image_size)
    
    loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True,
        pin_memory=True if device.type == "cuda" else False
    )
    
    logger.info(f"Dataset cargado: {len(train_ds):,} imágenes, {len(loader):,} batches")
    
    # Inicializar modelos
    logger.info("Inicializando modelos...")
    netG = Generator(args.z_dim, img_channels=3, feature_g=args.g_channels, out_size=args.image_size)
    
    if args.disc.lower() == "dcgan":
        netD = DiscriminatorDCGAN(img_channels=3, feature_d=args.d_channels, dropout_rate=args.dropout)
        logger.info("Usando discriminador DCGAN clásico")
    else:
        netD = DiscriminatorPretrained(
            args.disc, 
            pretrained=not args.no_pretrained, 
            dropout_rate=args.dropout
        )
        logger.info(f"Usando discriminador pretrained: {args.disc}")
    
    # Inicializar pesos
    netG.apply(weights_init_dcgan)
    if isinstance(netD, DiscriminatorDCGAN):
        netD.apply(weights_init_dcgan)
    else:
        # Solo inicializar el clasificador para modelos pretrained
        netD.classifier.apply(weights_init_xavier)
    
    netG.to(device)
    netD.to(device)
    
    # Información de modelos
    g_params = count_parameters(netG)
    d_params = count_parameters(netD)
    logger.info(f"Parámetros - Generador: {g_params:,}, Discriminador: {d_params:,}, Total: {g_params + d_params:,}")
    
    # Configurar optimizadores
    criterion = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(
        netG.parameters(), 
        lr=args.lr_g, 
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    optD = torch.optim.Adam(
        netD.parameters(), 
        lr=args.lr_d, 
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Schedulers
    if args.use_scheduler:
        scheduler_G = torch.optim.lr_scheduler.StepLR(optG, step_size=args.epochs//3, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optD, step_size=args.epochs//3, gamma=0.5)
    else:
        scheduler_G = scheduler_D = None
    
    # Ruido fijo para evaluación
    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)
    
    # Configurar etiquetas
    real_label_val = args.real_label_smooth if args.label_smoothing else 1.0
    fake_label_val = args.fake_label_smooth if args.label_smoothing else 0.0
    
    logger.info(f"Etiquetas: real={real_label_val}, fake={fake_label_val}")
    
    # Métricas
    metrics = TrainingMetrics()
    
    # Cargar checkpoint si existe
    start_epoch = 1
    if args.resume:
        resume_path = ckpt_dir / "latest.pt"
        if resume_path.exists():
            logger.info(f"Resumiendo desde checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            netG.load_state_dict(checkpoint['netG'])
            netD.load_state_dict(checkpoint['netD'])
            optG.load_state_dict(checkpoint['optG'])
            optD.load_state_dict(checkpoint['optD'])
            start_epoch = checkpoint['epoch'] + 1
            if 'metrics' in checkpoint:
                # Restaurar métricas si están disponibles
                for key, value in checkpoint['metrics'].items():
                    if hasattr(metrics, key):
                        setattr(metrics, key, value)
            logger.info(f"Resumiendo desde época {start_epoch}")
    
    # Loop de entrenamiento
    logger.info("Iniciando loop de entrenamiento...")
    iteration = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        netG.train()
        netD.train()
        
        # Barra de progreso
        if HAS_TQDM:
            pbar = tqdm(loader, desc=f"Época {epoch}/{args.epochs}", leave=False)
        else:
            pbar = loader
        
        epoch_losses_D = []
        epoch_losses_G = []
        
        for i, (imgs, _) in enumerate(pbar):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            
            # Agregar ruido para regularización
            if args.noise_std > 0:
                imgs = add_noise(imgs, args.noise_std)
            
            # =====================================
            # Entrenar Discriminador
            # =====================================
            optD.zero_grad()
            
            # Batch real
            real_labels = torch.full((batch_size,), real_label_val, device=device)
            if args.label_noise > 0:
                real_labels += torch.randn_like(real_labels) * args.label_noise
            
            output_real = netD(imgs)
            loss_D_real = criterion(output_real, real_labels)
            
            # Batch falso
            noise = torch.randn(batch_size, args.z_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            fake_labels = torch.full((batch_size,), fake_label_val, device=device)
            if args.label_noise > 0:
                fake_labels += torch.randn_like(fake_labels) * args.label_noise
            
            output_fake = netD(fake_imgs.detach())
            loss_D_fake = criterion(output_fake, fake_labels)
            
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            
            # Gradient clipping opcional
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(netD.parameters(), args.grad_clip)
            
            optD.step()
            
            # =====================================
            # Entrenar Generador
            # =====================================
            optG.zero_grad()
            
            output_fake_for_G = netD(fake_imgs)
            real_labels_for_G = torch.full((batch_size,), real_label_val, device=device)
            loss_G = criterion(output_fake_for_G, real_labels_for_G)
            
            loss_G.backward()
            
            # Gradient clipping opcional
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(netG.parameters(), args.grad_clip)
            
            optG.step()
            
            # Actualizar métricas
            epoch_losses_D.append(loss_D.item())
            epoch_losses_G.append(loss_G.item())
            
            # Logging detallado
            if iteration % args.log_every == 0:
                with torch.no_grad():
                    real_score = torch.sigmoid(output_real).mean().item()
                    fake_score = torch.sigmoid(output_fake).mean().item()
                    
                    # Calcular normas de gradientes
                    grad_norm_G = calculate_gradient_norm(netG)
                    grad_norm_D = calculate_gradient_norm(netD)
                    
                    metrics.update(
                        loss_D.item(), loss_G.item(),
                        real_score, fake_score,
                        grad_norm_G=grad_norm_G,
                        grad_norm_D=grad_norm_D
                    )
                
                if HAS_TQDM:
                    pbar.set_postfix({
                        'D_loss': f'{loss_D.item():.4f}',
                        'G_loss': f'{loss_G.item():.4f}',
                        'D(x)': f'{real_score:.3f}',
                        'D(G(z))': f'{fake_score:.3f}'
                    })
                else:
                    if iteration % (args.log_every * 5) == 0:  # Log menos frecuente sin tqdm
                        logger.info(f"[{epoch:03d}/{args.epochs}] it {iteration:06d} | "
                                  f"lossD {loss_D.item():.4f} | lossG {loss_G.item():.4f} | "
                                  f"D(x) {real_score:.3f} | D(G(z)) {fake_score:.3f}")
            
            iteration += 1
        
        # Finalizar época
        epoch_time = time.time() - epoch_start
        metrics.add_epoch_time(epoch_time)
        
        if scheduler_G:
            metrics.lr_G.append(scheduler_G.get_last_lr()[0])
            metrics.lr_D.append(scheduler_D.get_last_lr()[0])
            scheduler_G.step()
            scheduler_D.step()
        
        # Logging de época
        avg_loss_D = sum(epoch_losses_D) / len(epoch_losses_D) if epoch_losses_D else 0
        avg_loss_G = sum(epoch_losses_G) / len(epoch_losses_G) if epoch_losses_G else 0
        
        logger.info(f"Época {epoch}/{args.epochs} | "
                   f"D_loss: {avg_loss_D:.4f} | G_loss: {avg_loss_G:.4f} | "
                   f"Tiempo: {epoch_time:.2f}s")
        
        # Generar y guardar muestras
        with torch.no_grad():
            netG.eval()
            fake_fixed = netG(fixed_noise).cpu()
            grid = denorm(fake_fixed)
            vutils.save_image(grid, samples_dir / f"epoch_{epoch:03d}.png", nrow=8, padding=2)
        
        # Guardar checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = {
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "args": vars(args),
                "metrics": {
                    'losses_D': metrics.losses_D,
                    'losses_G': metrics.losses_G,
                    'scores_real': metrics.scores_real,
                    'scores_fake': metrics.scores_fake,
                    'lr_G': metrics.lr_G,
                    'lr_D': metrics.lr_D,
                    'epoch_times': metrics.epoch_times,
                    'gradient_norms_G': metrics.gradient_norms_G,
                    'gradient_norms_D': metrics.gradient_norms_D
                }
            }
            save_checkpoint(checkpoint, ckpt_dir, f"epoch_{epoch:03d}.pt")
            logger.info(f"Checkpoint guardado: epoch_{epoch:03d}.pt")
        
        # Plotear progreso
        if args.plot_every > 0 and epoch % args.plot_every == 0:
            metrics.plot_metrics(logs_dir / f"metrics_epoch_{epoch:03d}.png")
    
    # Finalizar entrenamiento
    total_time = time.time() - start_time
    logger.info(f"Entrenamiento completado en {total_time:.2f}s ({total_time/60:.2f} minutos)")
    
    # Guardar métricas finales
    metrics.save_to_json(logs_dir / "final_metrics.json")
    metrics.plot_metrics(logs_dir / "final_metrics.png")
    
    logger.info(f"Resultados guardados en: {out_dir}")
    logger.info(f"- Muestras: {samples_dir}")
    logger.info(f"- Checkpoints: {ckpt_dir}")
    logger.info(f"- Logs: {logs_dir}")

def parse_args():
    """Parser de argumentos mejorado con más opciones"""
    p = argparse.ArgumentParser(
        description="DCGAN con discriminador preentrenado - Entrenamiento mejorado",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset y directorios
    p.add_argument("--dataset", type=str, default="fashion_mnist", 
                   choices=["fashion_mnist", "mnist", "kmnist", "cifar10"],
                   help="Dataset a usar")
    p.add_argument("--data-dir", type=str, default="./data",
                   help="Directorio para datos")
    p.add_argument("--out-dir", type=str, default="./runs/dcgan_pretrained",
                   help="Directorio de salida")
    
    # Entrenamiento
    p.add_argument("--epochs", type=int, default=20,
                   help="Número de épocas")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Tamaño de batch")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Número de workers para DataLoader")
    
    # Arquitectura
    p.add_argument("--z-dim", type=int, default=100,
                   help="Dimensión del espacio latente")
    p.add_argument("--image-size", type=int, default=64,
                   help="Tamaño de imagen (asume cuadrada)")
    p.add_argument("--g-channels", type=int, default=64,
                   help="Número de canales base del generador")
    p.add_argument("--d-channels", type=int, default=64,
                   help="Número de canales base del discriminador")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="Tasa de dropout para regularización")
    
    # Discriminador
    p.add_argument("--disc", type=str, default="resnet18",
                   choices=["dcgan", "resnet18", "mobilenet_v2", "efficientnet_b0", "vgg11_bn"],
                   help="Tipo de discriminador")
    p.add_argument("--no-pretrained", action="store_true",
                   help="NO usar pesos pretrained (inicialización aleatoria)")
    
    # Optimización
    p.add_argument("--lr-g", type=float, default=0.0002,
                   help="Learning rate del generador")
    p.add_argument("--lr-d", type=float, default=0.0002,
                   help="Learning rate del discriminador")
    p.add_argument("--beta1", type=float, default=0.5,
                   help="Beta1 para Adam")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="Beta2 para Adam")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="Weight decay para regularización")
    p.add_argument("--use-scheduler", action="store_true",
                   help="Usar scheduler de learning rate")
    
    # Regularización y estabilidad
    p.add_argument("--label-smoothing", action="store_true",
                   help="Usar label smoothing")
    p.add_argument("--real-label-smooth", type=float, default=0.9,
                   help="Valor para etiquetas reales con smoothing")
    p.add_argument("--fake-label-smooth", type=float, default=0.1,
                   help="Valor para etiquetas falsas con smoothing")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Desviación estándar del ruido en inputs del discriminador")
    p.add_argument("--label-noise", type=float, default=0.0,
                   help="Desviación estándar del ruido en etiquetas")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Gradient clipping (0 para desactivar)")
    
    # Logging y guardado
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para reproducibilidad")
    p.add_argument("--cpu", action="store_true",
                   help="Forzar uso de CPU aunque CUDA esté disponible")
    p.add_argument("--log-every", type=int, default=100,
                   help="Frecuencia de logging (iteraciones)")
    p.add_argument("--save-every", type=int, default=5,
                   help="Frecuencia de guardado de checkpoints (épocas)")
    p.add_argument("--plot-every", type=int, default=5,
                   help="Frecuencia de plotting (épocas, 0 para desactivar)")
    p.add_argument("--resume", action="store_true",
                   help="Resumir entrenamiento desde último checkpoint")
    
    # Evaluación
    p.add_argument("--eval-only", action="store_true",
                   help="Solo evaluar (cargar checkpoint y generar muestras)")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Ruta al checkpoint para evaluación")
    p.add_argument("--num-samples", type=int, default=64,
                   help="Número de muestras a generar en evaluación")
    
    return p.parse_args()

def evaluate_model(args):
    """Función para evaluar modelo sin entrenar"""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"🔍 Evaluando modelo en dispositivo: {device}")
    
    # Cargar checkpoint
    if not args.checkpoint:
        checkpoint_path = Path(args.out_dir) / "checkpoints" / "latest.pt"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró checkpoint en: {checkpoint_path}")
    
    print(f"📂 Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Obtener configuración del checkpoint
    saved_args = checkpoint.get('args', vars(args))
    
    # Inicializar modelos
    netG = Generator(
        saved_args.get('z_dim', args.z_dim),
        img_channels=3,
        feature_g=saved_args.get('g_channels', args.g_channels),
        out_size=saved_args.get('image_size', args.image_size)
    ).to(device)
    
    netG.load_state_dict(checkpoint['netG'])
    netG.eval()
    
    print(f"✅ Modelo cargado desde época {checkpoint.get('epoch', 'desconocida')}")
    
    # Crear directorio de evaluación
    eval_dir = Path(args.out_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar muestras diversas
    print(f"🎨 Generando {args.num_samples} muestras...")
    
    with torch.no_grad():
        z = torch.randn(args.num_samples, saved_args.get('z_dim', args.z_dim), 1, 1, device=device)
        fake_samples = netG(z).cpu()
        grid = denorm(fake_samples)
        
        # Guardar grid de muestras
        vutils.save_image(grid, eval_dir / "generated_samples.png", nrow=8, padding=2)
        
        # Guardar muestras individuales
        individual_dir = eval_dir / "individual_samples"
        individual_dir.mkdir(exist_ok=True)
        
        for i in range(min(16, args.num_samples)):  # Guardar primeras 16 como individuales
            sample = grid[i]
            vutils.save_image(sample, individual_dir / f"sample_{i+1:03d}.png")
    
    print(f"💾 Muestras guardadas en: {eval_dir}")
    print(f"   - Grid completo: {eval_dir / 'generated_samples.png'}")
    print(f"   - Muestras individuales: {individual_dir}")

if __name__ == "__main__":
    args = parse_args()
    
    # Validar configuración
    if args.eval_only:
        evaluate_model(args)
    else:
        # Crear directorio de salida
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Guardar configuración
        config_path = Path(args.out_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        print(f"📋 Configuración guardada en: {config_path}")
        print(f"🚀 Iniciando entrenamiento...")
        print("=" * 80)
        
        # Entrenar
        train(args)
