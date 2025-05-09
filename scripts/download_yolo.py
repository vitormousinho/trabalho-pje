import os
import urllib.request
import shutil

def download_file(url, output_file):
    """Baixa um arquivo da web."""
    print(f"Baixando {url}...")
    with urllib.request.urlopen(url) as response, open(output_file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Arquivo salvo como {output_file}")

def main():
    """Função principal."""
    # Caminho ajustado para seu repositório
    project_dir = r"C:\Users\vitor\projeto pje\trabalho-pje"
    models_dir = os.path.join(project_dir, "models", "yolo_weights")
    os.makedirs(models_dir, exist_ok=True)
    
    # URLs para download
    yolo_cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
    coco_names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    
    # Caminhos de destino
    yolo_cfg_path = os.path.join(models_dir, "yolov4.cfg")
    coco_names_path = os.path.join(models_dir, "coco.names")
    yolo_weights_path = os.path.join(models_dir, "yolov4.weights")
    
    # Baixar arquivos menores
    download_file(yolo_cfg_url, yolo_cfg_path)
    download_file(coco_names_url, coco_names_path)
    
    # Instruções para baixar os pesos
    print("\nAVISO: Os pesos do YOLOv4 são um arquivo grande (245MB) e precisam ser baixados manualmente.")
    print("Por favor, visite: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
    print(f"E salve o arquivo em: {yolo_weights_path}")

if __name__ == "__main__":
    main()