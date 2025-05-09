#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controlador para gerenciar as câmeras de monitoramento.
"""

import cv2
import logging
import threading
import time
import numpy as np

logger = logging.getLogger(__name__)

class CameraController:
    """Classe para gerenciar múltiplas câmeras de monitoramento."""
    
    def __init__(self, config):
        """
        Inicializa o controlador de câmeras.
        
        Args:
            config: Dicionário com configurações das câmeras
        """
        self.config = config
        self.cameras = {}
        self.frames = {}
        self.running = False
        self.lock = threading.Lock()
        
        # Inicializar as câmeras
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Inicializa as câmeras conforme a configuração."""
        try:
            for camera_id, camera_config in self.config['cameras'].items():
                source = camera_config['source']
                
                # O source pode ser um índice (int) ou um URL (string)
                logger.info(f"Inicializando câmera {camera_id} com source {source}")
                
                try:
                    # Tentar inicializar a câmera
                    cap = cv2.VideoCapture(source)
                    
                    if not cap.isOpened():
                        logger.warning(f"Não foi possível abrir a câmera {camera_id}")
                        # Em caso de falha, usar uma imagem preta como fallback
                        self.cameras[camera_id] = None
                        self.frames[camera_id] = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        self.cameras[camera_id] = cap
                        # Capturar o primeiro frame
                        ret, frame = cap.read()
                        if ret:
                            self.frames[camera_id] = frame
                        else:
                            self.frames[camera_id] = np.zeros((480, 640, 3), dtype=np.uint8)
                
                except Exception as e:
                    logger.error(f"Erro ao inicializar câmera {camera_id}: {e}")
                    self.cameras[camera_id] = None
                    self.frames[camera_id] = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Iniciar thread de captura contínua
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
        except Exception as e:
            logger.error(f"Erro ao inicializar câmeras: {e}")
            raise
    
    def _capture_loop(self):
        """Loop de captura contínua que é executado em uma thread separada."""
        while self.running:
            for camera_id, cap in self.cameras.items():
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        with self.lock:
                            self.frames[camera_id] = frame
            time.sleep(0.03)  # ~30 FPS
    
    def capture_frames(self):
        """
        Captura frames de todas as câmeras.
        
        Returns:
            Dicionário com os frames capturados para cada câmera
        """
        with self.lock:
            # Retornar uma cópia para evitar modificações externas
            return {camera_id: frame.copy() for camera_id, frame in self.frames.items()}
    
    def release(self):
        """Libera os recursos das câmeras."""
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
            
        for camera_id, cap in self.cameras.items():
            if cap is not None:
                cap.release()
        
        logger.info("Todas as câmeras foram liberadas")