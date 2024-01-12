# Transmision Multi-Usuarios Usando Tecnicas de Acceso Multiple Usando Portadoras Audibles

Este repositorio cuenta con la implementacion de un sistema de 'Transmision Multi-Usuarios Usando Tecnicas de Acceso Multiple Usando Portadoras Audibles'.
Para su correcta ejecucio es necesario instalar las librerias utilizadas junto a la version de python=3.9. Para esto se debe ejecturar el codigo:

`$ pip install -r requirements.txt`

Posteriormente es necesario al menos 2 computadores. Uno de estos tiene el rol de ser Receiver. Por lo que se debe ejecutar el notebook 
`Demo.ipynb` y seguir las intruciones dadas en este.

En el caso del resto de computadores, estos deben tomar el rol de Sender, para lo cual deben ejectutar el comando send_data. Luego, se debe editar el archivo 
`configs.yaml` de acuerdo a las caracteristicas que se le quiere dar a la señal enviada. Es importante notar que el parametro `N` determina la imagen y texto enviado. 
Si este tiene un valor 1 se envia la primera imagen y el primer texto y en el caso de ser 2 se envia la segunda imagen y el segundo texto.

Por ultimo se adjunta un archivo `Grabacion.ipynb` que contiene una ejecucion de prueba que se basa en una grabacion realizada previamente y que corresponde a la mejor
ejecucion obtenida.
