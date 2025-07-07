import asyncio
import websockets
import serial
import serial.tools.list_ports
import logging
from datetime import datetime
import platform

class ArduinoWebSocketBridge:
    def __init__(self):
        self.serial_port = None
        self.clients = set()
        self.is_running = True
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'joystick_bridge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def init_serial(self):
        """Inicializa la conexión serial con Arduino"""
        while self.serial_port is None and self.is_running:
            try:
                # Buscar puertos Arduino
                arduino_ports = [
                    p.device for p in serial.tools.list_ports.comports()
                    if 'Arduino' in p.description or 'CH340' in p.description
                ]
                
                if arduino_ports:
                    # En Windows, no usar exclusive=False
                    serial_config = {
                        'port': arduino_ports[0],
                        'baudrate': 9600,
                        'timeout': 0.1
                    }
                    
                    self.serial_port = serial.Serial(**serial_config)
                    
                    # Limpiar buffer
                    self.serial_port.reset_input_buffer()
                    print(f"\n{'='*50}")
                    print(f"Arduino conectado en {arduino_ports[0]}")
                    print(f"{'='*50}\n")
                    self.logger.info(f"Arduino conectado en {arduino_ports[0]}")
                else:
                    print("\nBuscando Arduino... No encontrado.")
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"\nError al conectar con Arduino: {e}")
                await asyncio.sleep(2)

    async def handle_client(self, websocket):
        """Maneja la conexión con Unity"""
        self.clients.add(websocket)
        client_info = websocket.remote_address
        print(f"\nUnity conectado desde {client_info}")
        self.logger.info(f"Unity conectado desde {client_info}")
        
        try:
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            print(f"\nUnity desconectado: {client_info}")
            self.logger.info(f"Unity desconectado: {client_info}")
        finally:
            self.clients.remove(websocket)

    async def broadcast_serial_data(self):
        """Lee datos de Arduino y los transmite a Unity"""
        last_data = None
        error_count = 0
        max_errors = 3
        
        while self.is_running:
            if self.serial_port and self.serial_port.is_open:
                try:
                    if self.serial_port.in_waiting:
                        data = self.serial_port.readline().decode().strip()
                        if data and data != last_data:
                            try:
                                # Verificar formato de datos
                                angle, magnitude = map(float, data.split(','))
                                # Solo enviar si los valores son válidos
                                if 0 <= angle <= 360 and 0 <= magnitude <= 100:
                                    print(f"\rJoystick -> Ángulo: {angle:>6.1f}° | Magnitud: {magnitude:>5.1f}% | Clientes: {len(self.clients)}", end="")
                                    if self.clients:
                                        await asyncio.gather(*[
                                            client.send(data) for client in self.clients
                                        ])
                                    last_data = data
                                    error_count = 0
                            except ValueError:
                                error_count += 1
                                if error_count >= max_errors:
                                    print(f"\nError: Datos inválidos recibidos: {data}")
                                    error_count = 0
                except Exception as e:
                    print(f"\nError leyendo datos seriales: {e}")
                    self.serial_port = None
                    await self.init_serial()
            else:
                await self.init_serial()
            await asyncio.sleep(0.01)

    async def start_server(self):
        """Inicia el servidor WebSocket"""
        port = 8080
        server = await websockets.serve(
            self.handle_client, 
            "localhost", 
            port,
            ping_interval=None
        )
        
        print(f"\nServidor WebSocket iniciado en ws://localhost:{port}")
        print("Esperando conexión de Unity...")
        self.logger.info(f"Servidor WebSocket iniciado en puerto {port}")
        
        try:
            await self.broadcast_serial_data()
        except Exception as e:
            print(f"\nError en el servidor: {e}")
            self.logger.error(f"Error en el servidor: {e}")
        finally:
            server.close()
            await server.wait_closed()

    def run(self):
        """Inicia el programa puente"""
        print("\n=== Programa Puente Arduino-Unity ===")
        print("Iniciando...")
        
        try:
            asyncio.run(self.start_server())
        except KeyboardInterrupt:
            print("\n\nPrograma terminado por el usuario")
        except Exception as e:
            print(f"\nError inesperado: {e}")
        finally:
            self.is_running = False
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            self.logger.info("Programa terminado")

if __name__ == "__main__":
    bridge = ArduinoWebSocketBridge()
    bridge.run()