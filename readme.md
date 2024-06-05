# hola
## instalacion
 - instalar conda // anaconda // miniconda
 - activar el entorno virtual
 - instalar la GPU `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0` https://www.tensorflow.org/install/pip?hl=es#windows-native 
 - probar la GPU `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
 - instalar el compilador de c++ https://visualstudio.microsoft.com/es/visual-cpp-build-tools/ descargar built tools y seleccionar la opcion de  desarrollo de escritorio c++ e instalar
 - instalar git en conda `conda install git`
 instalar el requierements.txt `pip install -r requirements.txt`
 - en caso de usar jupyter en vscode hay que instalar el kernel para ejecutar `conda install -n ar ipykernel --update-deps --force-reinstall`