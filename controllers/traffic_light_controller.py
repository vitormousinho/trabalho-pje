#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controlador para gerenciar os semáforos.
"""

import time
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class LightState(Enum):
    """Enumeração para os estados do semáforo."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"

class TrafficLightController:
    """Classe para gerenciar os semáforos."""
    
    def __init__(self, config):
        """
        Inicializa o controlador de semáforos.
        
        Args:
            config: Dicionário com configurações dos semáforos
        """
        self.config = config
        
        # Definir estados iniciais
        self.directions = ["north", "east", "south", "west"]
        self.current_state = {direction: LightState.RED for direction in self.directions}
        self.current_state["north"] = LightState.GREEN  # Norte começa com verde
        
        # Tempos padrão (em segundos)
        self.default_green_time = config.get("default_green_time", 30)
        self.min_green_time = config.get("min_green_time", 10)
        self.max_green_time = config.get("max_green_time", 90)
        self.yellow_time = config.get("yellow_time", 3)
        
        # Controle de tempo
        self.last_change_time = time.time()
        self.current_green_direction = "north"
        self.green_duration = self.default_green_time
        
        # Thread para controle autônomo dos semáforos
        self.running = True
        self.auto_mode = True
        self.lock = threading.Lock()
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("Controlador de semáforos iniciado")
    
    def _control_loop(self):
        """Loop de controle automático dos semáforos."""
        while self.running:
            if self.auto_mode:
                current_time = time.time()
                elapsed_time = current_time - self.last_change_time
                
                with self.lock:
                    # Verificar se é hora de mudar o sinal
                    if self.current_state[self.current_green_direction] == LightState.GREEN:
                        if elapsed_time >= self.green_duration:
                            self._change_to_yellow(self.current_green_direction)
                            self.last_change_time = current_time
                    
                    elif self.current_state[self.current_green_direction] == LightState.YELLOW:
                        if elapsed_time >= self.yellow_time:
                            # Determinar a próxima direção (sentido horário)
                            next_index = (self.directions.index(self.current_green_direction) + 1) % len(self.directions)
                            next_direction = self.directions[next_index]
                            
                            self._change_to_red(self.current_green_direction)
                            self._change_to_green(next_direction)
                            
                            self.current_green_direction = next_direction
                            self.green_duration = self.default_green_time
                            self.last_change_time = current_time
            
            time.sleep(0.1)
    
    def _change_to_green(self, direction):
        """Muda o semáforo para verde na direção especificada."""
        self.current_state[direction] = LightState.GREEN
        logger.info(f"Semáforo {direction}: VERDE")
        # Aqui você adicionaria o código para controlar o hardware real
    
    def _change_to_yellow(self, direction):
        """Muda o semáforo para amarelo na direção especificada."""
        self.current_state[direction] = LightState.YELLOW
        logger.info(f"Semáforo {direction}: AMARELO")
        # Aqui você adicionaria o código para controlar o hardware real
    
    def _change_to_red(self, direction):
        """Muda o semáforo para vermelho na direção especificada."""
        self.current_state[direction] = LightState.RED
        logger.info(f"Semáforo {direction}: VERMELHO")
        # Aqui você adicionaria o código para controlar o hardware real
    
    def update(self, decision):
        """
        Atualiza os semáforos com base na decisão do algoritmo.
        
        Args:
            decision: Dicionário com as decisões para cada direção.
                     Ex: {"direction": "north", "green_time": 45}
        """
        with self.lock:
            # Desativar modo automático temporariamente
            self.auto_mode = False
            
            direction = decision.get("direction")
            green_time = decision.get("green_time", self.default_green_time)
            
            # Validar o tempo verde dentro dos limites
            green_time = max(min(green_time, self.max_green_time), self.min_green_time)
            
            # Se a direção atual já estiver verde, apenas ajustar o tempo
            if direction == self.current_green_direction and self.current_state[direction] == LightState.GREEN:
                self.green_duration = green_time
                logger.info(f"Ajustando tempo verde para {direction}: {green_time}s")
                
            # Caso contrário, mudar o semáforo atual para amarelo
            else:
                # Primeiro, mudar o atual para amarelo (se não estiver vermelho)
                if self.current_state[self.current_green_direction] == LightState.GREEN:
                    self._change_to_yellow(self.current_green_direction)
                    
                    # Aguardar o tempo do amarelo
                    time.sleep(self.yellow_time)
                    
                    # Mudar para vermelho
                    self._change_to_red(self.current_green_direction)
                
                # Mudar todos os outros para vermelho, exceto o novo
                for d in self.directions:
                    if d != direction and self.current_state[d] != LightState.RED:
                        self._change_to_red(d)
                
                # Mudar o novo para verde
                self._change_to_green(direction)
                self.current_green_direction = direction
                self.green_duration = green_time
                self.last_change_time = time.time()
                
                logger.info(f"Mudando para verde: {direction} por {green_time}s")
            
            # Reativar modo automático após a atualização
            self.auto_mode = True
    
    def reset(self):
        """Reseta os semáforos para o estado padrão."""
        self.running = False
        if hasattr(self, 'control_thread'):
            self.control_thread.join(timeout=1.0)
            
        # Colocar todos em vermelho
        for direction in self.directions:
            self._change_to_red(direction)
            
        logger.info("Semáforos resetados")