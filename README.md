# Projeto_Final_Redes_Neurais

Proyecto para entrenamiento y evaluación de modelos BERT en tareas de Question Answering (SQuAD) en portugués.



Descripción
- Este proyecto contiene experimentos de redes neuronales aplicados a la tarea de Question Answering en portugués usando modelos BERTimbau (base y large).
- Se exploran técnicas de ajuste eficiente de parámetros como LoRA y QLoRA, así como cuantización y pruebas con diferentes learning rates.
- Incluye scripts para preprocesamiento, postprocesamiento, entrenamiento y evaluación, permitiendo comparar el impacto de la cuantización y el ajuste de hiperparámetros en el rendimiento y eficiencia de los modelos.


Estructura principal
- `qa_bertimbau/`: subproyecto con implementaciones para:
	- `bertimbau_base/` y `bertimbau_large/`: scripts para entrenamiento y evaluación (`main.py`, `main_qlora.py`, `main_lora.py`, etc.), junto con carpetas `data/` y `results/`.
	- `tucano_base/`: scripts y datos para experimentos adicionales.
- `data/`: ubicación esperada para archivos JSON de entrenamiento/validación.

Cómo empezar (rápido)
1. Crear y activar un entorno virtual (recomendado con conda):
	```sh
	python -m venv .venv
	.venv/bin/activate
	```
	o con conda:
	```sh
	conda create -n qa_squad python=3.10 -y
	conda activate qa_squad
	```
2. Instalar dependencias (puedes revisar `qa_bertimbau/requirements.txt`):
	```sh
	pip install -r qa_bertimbau/requirements.txt
	```
3. Ejecutar un ejemplo de entrenamiento (ajusta batch sizes y nombres según tu GPU):
	```sh
	cd qa_bertimbau/bertimbau_base
	python main_qlora.py
	```


Notas importantes
- Los artefactos pesados (modelos, checkpoints) están excluidos por `.gitignore`. Evita subir archivos grandes como `*.pt`, `*.safetensors`, `runs/`, `results/`.
- Para compartir modelos entrenados, usa servicios externos como Hugging Face Hub, Google Drive o S3.
- El subproyecto `qa_bertimbau` fue convertido a carpeta normal dentro del repo principal.


Contribuir
- Puedes abrir issues o pull requests en GitHub para sugerencias o mejoras.
- Antes de commitear, ejecuta linters y pruebas si están disponibles.


Contacto
- Maintainer: `MarielaNina` (https://github.com/MarielaNina)


Licencia
- Puedes añadir una licencia (MIT, Apache, etc.) si lo deseas. Solicítala y la agrego por ti.
