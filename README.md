# pageDetector
____
This is a telegram bot for document segmentation on the image.

In this project, the `MobileNetv3` neural network was used to obtain the document mask on the original image. The model was trained on an artificially generated dataset.

After obtaining the document mask, post-processing is used to transform the document to the size of the original image. This algorithm is implemented in the extractPage.py.

The `aiogram` framework was used to interact with the Telegram Bot API.

-----
## Installation and launch
You can start it with `Docker`:
```Docker
docker pull ivansergeevphysics/page_detector:1.4
docker run -d -e BOT_TOKEN= 'your token' ivansergeevphysics/page_detector:1.4
```
_____
Or you can run it locally by cloning this repository and running:
```python
pip install -r requirements.txt
export BOT_TOKEN='your token'
python bot.py
```
___
### Demo:
<img style="display: block; 
           margin-left: auto;
           margin-right: auto;"
           src="https://github.com/IvanSergeevPhysics/pageDetector/blob/9f57e2c510540b9d11a844be11576d6237e127ee/demo.gif" height="600">
</img>
