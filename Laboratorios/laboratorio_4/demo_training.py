#!/usr/bin/env python3
"""
Script de demostración para DCGAN con discriminador preentrenado
================================================================

Este script muestra diferentes configuraciones de entrenamiento
para demostrar las capacidades del sistema mejorado.

Ejecuta diferentes experimentos con distintos discriminadores
y configuraciones para comparar resultados.
"""

import subprocess
import time
import sys
from pathlib import Path

def run_experiment(name, args, description):
    """Ejecutar un experimento de entrenamiento"""
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENTO: {name}")
    print(f"📝 {description}")
    print(f"⚙️  Argumentos: {' '.join(args)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Ejecutar comando
        cmd = [sys.executable, "train_dcgan_pretrained.py"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"✅ {name} completado en {elapsed:.2f} segundos")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {name} falló después de {elapsed:.2f} segundos")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Función principal de demostración"""
    print("🚀 DCGAN - Suite de Experimentos de Demostración")
    print("=" * 60)
    
    # Lista de experimentos
    experiments = [
        {
            "name": "Baseline DCGAN",
            "args": [
                "--epochs", "3",
                "--disc", "dcgan", 
                "--batch-size", "64",
                "--out-dir", "./runs/demo_dcgan_baseline",
                "--log-every", "50"
            ],
            "description": "Discriminador DCGAN clásico con configuración básica"
        },
        {
            "name": "ResNet18 Pretrained",
            "args": [
                "--epochs", "3",
                "--disc", "resnet18",
                "--batch-size", "64", 
                "--label-smoothing",
                "--dropout", "0.3",
                "--out-dir", "./runs/demo_resnet18",
                "--log-every", "50"
            ],
            "description": "ResNet18 preentrenado con regularización mejorada"
        },
        {
            "name": "MobileNet Efficient",
            "args": [
                "--epochs", "3",
                "--disc", "mobilenet_v2",
                "--batch-size", "128",
                "--lr-g", "0.0001",
                "--lr-d", "0.0002", 
                "--label-smoothing",
                "--use-scheduler",
                "--out-dir", "./runs/demo_mobilenet",
                "--log-every", "50"
            ],
            "description": "MobileNetV2 con learning rates diferenciados y scheduler"
        },
        {
            "name": "MNIST Comparison",
            "args": [
                "--dataset", "mnist",
                "--epochs", "3",
                "--disc", "resnet18",
                "--batch-size", "64",
                "--label-smoothing",
                "--noise-std", "0.05",
                "--out-dir", "./runs/demo_mnist",
                "--log-every", "50"
            ],
            "description": "Experimento con dataset MNIST y ruido para regularización"
        },
        {
            "name": "High Quality Setup",
            "args": [
                "--epochs", "5",
                "--disc", "efficientnet_b0",
                "--batch-size", "32",
                "--lr-g", "0.0001",
                "--lr-d", "0.0002",
                "--label-smoothing",
                "--real-label-smooth", "0.95",
                "--fake-label-smooth", "0.05",
                "--dropout", "0.4",
                "--noise-std", "0.1",
                "--label-noise", "0.02",
                "--weight-decay", "1e-4",
                "--use-scheduler",
                "--out-dir", "./runs/demo_high_quality",
                "--log-every", "25"
            ],
            "description": "Configuración de alta calidad con máxima regularización"
        }
    ]
    
    # Ejecutar experimentos
    results = {}
    total_start = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n🔄 Ejecutando experimento {i}/{len(experiments)}")
        
        success = run_experiment(
            exp["name"], 
            exp["args"], 
            exp["description"]
        )
        
        results[exp["name"]] = success
        
        # Pequeña pausa entre experimentos
        if i < len(experiments):
            print("⏳ Pausa de 5 segundos...")
            time.sleep(5)
    
    # Resumen final
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE EXPERIMENTOS")
    print(f"{'='*60}")
    print(f"⏱️  Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
    print(f"📈 Experimentos completados:")
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    successful = sum(results.values())
    print(f"\n🎯 Tasa de éxito: {successful}/{len(experiments)} ({successful/len(experiments)*100:.1f}%)")
    
    # Mostrar ubicaciones de resultados
    print(f"\n📁 Resultados guardados en:")
    print(f"   📂 ./runs/demo_*")
    print(f"   🖼️  Muestras: ./runs/demo_*/samples/")
    print(f"   💾 Checkpoints: ./runs/demo_*/checkpoints/")
    print(f"   📊 Logs: ./runs/demo_*/logs/")
    
    print(f"\n💡 Para comparar resultados, revisa las imágenes generadas:")
    print(f"   - Baseline DCGAN: ./runs/demo_dcgan_baseline/samples/")
    print(f"   - ResNet18: ./runs/demo_resnet18/samples/")
    print(f"   - MobileNet: ./runs/demo_mobilenet/samples/")
    print(f"   - MNIST: ./runs/demo_mnist/samples/")
    print(f"   - Alta calidad: ./runs/demo_high_quality/samples/")
    
    print(f"\n🔍 Para evaluar modelos entrenados:")
    print(f"   python train_dcgan_pretrained.py --eval-only --checkpoint ./runs/demo_resnet18/checkpoints/latest.pt")
    
    print(f"\n🎉 ¡Demostración completada!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  Demostración interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error en la demostración: {e}")
        sys.exit(1)
