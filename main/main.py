#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de Monitoramento de Trânsito Inteligente
------------------------------------------------
Este programa monitora o fluxo de veículos através de câmeras e 
controla semáforos para otimizar o tráfego.
"""

import os
import time
import logging
from config.settings import Settings
from controllers.camera_controller import CameraController
from controllers.traffic_light_controller import TrafficLightController
from models.yolo_model import YOLOModel
from utils.traffic_analysis import TrafficAnalyzer

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("traffic_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Função principal que inicia o sistema de monitoramento."""
    logger.info("Iniciando Sistema de Monitoramento de Trânsito Inteligente")
    
    # Carregar configurações
    settings = Settings()
    
    # Inicializar o modelo YOLO
    model = YOLOModel(settings.model_path, settings.confidence_threshold)
    
    # Inicializar controlador de câmeras
    camera_controller = CameraController(settings.camera_config)
    
    # Inicializar analisador de tráfego
    traffic_analyzer = TrafficAnalyzer()
    
    # Inicializar controlador de semáforos
    traffic_light_controller = TrafficLightController(settings.traffic_light_config)
    
    try:
        while True:
            # Capturar frames da câmera
            frames = camera_controller.capture_frames()
            
            # Detectar veículos em cada frame
            detections = {}
            for camera_id, frame in frames.items():
                detections[camera_id] = model.detect(frame)
            
            # Analisar o fluxo de tráfego
            traffic_state = traffic_analyzer.analyze(detections)
            
            # Tomar decisão sobre os semáforos
            decision = traffic_analyzer.make_decision(traffic_state)
            
            # Atualizar os semáforos
            traffic_light_controller.update(decision)
            
            # Exibir estatísticas (opcional)
            logger.info(f"Estado do tráfego: {traffic_state}")
            logger.info(f"Decisão: {decision}")
            
            # Aguardar antes da próxima iteração
            time.sleep(settings.processing_interval)
            
    except KeyboardInterrupt:
        logger.info("Encerrando o sistema...")
    finally:
        # Limpeza e fechamento de recursos
        camera_controller.release()
        traffic_light_controller.reset()
        logger.info("Sistema encerrado")

if __name__ == "__main__":
    main()