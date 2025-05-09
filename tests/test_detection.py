#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes para o módulo de detecção de veículos.
"""

import os
import sys
import unittest
import cv2
import numpy as np

# Adicionar o diretório pai ao path para importar os módulos do projeto
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.yolo_model import YOLOModel
from config.settings import Settings

class TestDetection(unittest.TestCase):
    """Testes para a funcionalidade de detecção de veículos."""
    
    @classmethod
    def setUpClass(cls):
        """Configuração executada uma vez antes de todos os testes."""
        cls.settings = Settings()
        
        # Verificar se os arquivos do modelo existem
        model_path = cls.settings.model_path
        weights_path = os.path.join(model_path, "yolov4.weights")
        config_path = os.path.join(model_path, "yolov4.cfg")
        
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            cls.model = None
            cls.skip_tests = True
            print("Arquivos do modelo YOLO não encontrados. Pulando testes de detecção.")
        else:
            cls.model = YOLOModel(model_path, cls.settings.confidence_threshold)
            cls.skip_tests = False
    
    def setUp(self):
        """Configuração executada antes de cada teste."""
        if self.skip_tests:
            self.skipTest("Arquivos do modelo YOLO não encontrados")
    
    def test_detect_empty_image(self):
        """Testa detecção em uma imagem vazia."""
        # Criar uma imagem preta
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Executar detecção
        detections = self.model.detect(image)
        
        # Verificar resultado
        self.assertEqual(len(detections), 0, "Não deveria detectar nada em uma imagem vazia")
    
    def test_detect_with_invalid_input(self):
        """Testa comportamento com entrada inválida."""
        # Tentar com imagem de tamanho errado
        with self.assertRaises(Exception):
            self.model.detect(np.zeros((10, 10), dtype=np.uint8))

if __name__ == "__main__":
    unittest.main()