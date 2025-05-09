#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testes para os controladores do sistema.
"""

import os
import sys
import unittest
import time
from unittest.mock import MagicMock

# Adicionar o diretório pai ao path para importar os módulos do projeto
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from controllers.traffic_light_controller import TrafficLightController, LightState
from utils.traffic_analysis import TrafficAnalyzer

class TestTrafficLightController(unittest.TestCase):
    """Testes para o controlador de semáforos."""
    
    def setUp(self):
        """Configuração executada antes de cada teste."""
        # Configuração de teste
        self.config = {
            "default_green_time": 5,
            "min_green_time": 2,
            "max_green_time": 10,
            "yellow_time": 1
        }
        
        # Inicializar controlador
        self.controller = TrafficLightController(self.config)
        
        # Reduzir a variável running para False antes de cada teste
        self.controller.running = False
        if hasattr(self.controller, 'control_thread'):
            self.controller.control_thread.join(timeout=1.0)
    
    def test_initialization(self):
        """Testa a inicialização do controlador."""
        # Verificar estados iniciais
        self.assertEqual(self.controller.current_state["north"], LightState.GREEN)
        self.assertEqual(self.controller.current_state["east"], LightState.RED)
        self.assertEqual(self.controller.current_state["south"], LightState.RED)
        self.assertEqual(self.controller.current_state["west"], LightState.RED)
        
        # Verificar tempos configurados
        self.assertEqual(self.controller.default_green_time, 5)
        self.assertEqual(self.controller.min_green_time, 2)
        self.assertEqual(self.controller.max_green_time, 10)
        self.assertEqual(self.controller.yellow_time, 1)
    
    def test_update(self):
        """Testa a atualização do controlador."""
        # Testar mudança de direção
        decision = {"direction": "east", "green_time": 8}
        self.controller.update(decision)
        
        # Verificar se a direção mudou
        self.assertEqual(self.controller.current_green_direction, "east")
        self.assertEqual(self.controller.green_duration, 8)

class TestTrafficAnalyzer(unittest.TestCase):
    """Testes para o analisador de tráfego."""
    
    def setUp(self):
        """Configuração executada antes de cada teste."""
        self.analyzer = TrafficAnalyzer(congestion_threshold=10)
    
    def test_analyze(self):
        """Testa a análise de tráfego."""
        # Simular detecções
        detections = {
            "north": [["car", 0.9, [100, 100, 50, 50]]] * 5,  # 5 carros
            "east": [["car", 0.9, [100, 100, 50, 50]]] * 15,  # 15 carros (congestionado)
            "south": [["car", 0.9, [100, 100, 50, 50]]] * 2,  # 2 carros
            "west": [["car", 0.9, [100, 100, 50, 50]]] * 8    # 8 carros
        }
        
        # Analisar tráfego
        traffic_state = self.analyzer.analyze(detections)
        
        # Verificar resultados
        self.assertEqual(traffic_state["north"]["vehicle_count"], 5)
        self.assertEqual(traffic_state["east"]["vehicle_count"], 15)
        
        # Verificar nível de congestionamento
        self.assertTrue(traffic_state["east"]["congestion_level"] > 
                       traffic_state["north"]["congestion_level"])
    
    def test_make_decision(self):
        """Testa a tomada de decisão."""
        # Criar estado de tráfego simulado
        traffic_state = {
            "north": {"vehicle_count": 5, "avg_count": 5, "congestion_level": 0.5},
            "east": {"vehicle_count": 15, "avg_count": 15, "congestion_level": 1.0},
            "south": {"vehicle_count": 2, "avg_count": 2, "congestion_level": 0.2},
            "west": {"vehicle_count": 8, "avg_count": 8, "congestion_level": 0.8}
        }
        
        # Tomar decisão
        decision = self.analyzer.make_decision(traffic_state)
        
        # Verificar decisão
        self.assertEqual(decision["direction"], "east")  # Deve escolher a direção mais congestionada

if __name__ == "__main__":
    unittest.main()