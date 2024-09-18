import gradio as gr
import pyttsx3
import librosa
import argparse
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizer
from model.configuration_model import MAINconfig
from model.modeling import MAIN
from dataset.common import load_image, load_video
from dataset.processor.ImageCaption_processor import ImageCaptionProcessorWithoutCrop
from utils.bbox import draw_bbox_to_image
import speech_recognition
from icecream import ic


def audio2text(input_audio):
    # 语音识别
    recognizer = speech_recognition.Recognizer()
    # 从WAV文件加载音频数据
    with speech_recognition.AudioFile(input_audio) as source:
        audio_data = recognizer.record(source)
    # 尝试使用Google Web Speech API进行语音识别
    try:
        # 识别语音
        text = recognizer.recognize_google(audio_data)  # 根据需要设置语言
        print(f"识别结果: {text}")
    except speech_recognition.UnknownValueError:
        print("Google Web Speech API无法理解音频")
    except speech_recognition.RequestError as e:
        print(f"无法从Google Web Speech API获得结果; {e}")
    return text


class model_wrapper():
    def __init__(self,config: MAINconfig, model: MAIN, tokenizer: PreTrainedTokenizer) -> None:
        self.config = config
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

        self.image_processor = ImageCaptionProcessorWithoutCrop()

    def generate(self, pixel_values):
        args = {
            'max_new_tokens': 30,
            'num_beams': 5,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True,
        }

        output = self.model.generate(
            pixel_values = pixel_values,
            attention_mask = None,
            **args
        )
        return output['generated_ids'], output['logits_bbox']


    def tag_img(self,img: np.ndarray,input_text,input_audio):
        # 新增input_text,input_audio
        # 打印参数类型
        print(type(img))
        print(type(input_text), input_text)
        print(type(input_audio), input_audio)

        # audio2text
        if input_audio is not None:
            text_get = audio2text(input_audio)

        img = Image.fromarray(img).convert("RGB")
        img, text = self.image_processor(img, None)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(self.model.device)

        generated_ids, logits_bbox = self.generate(img)
        # 输出结果赋给变量text
        text = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]

        # 输出图片赋给变量output_img
        output_img = img

        # 文字转语音
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)

        engine.say(text)
        engine.save_to_file(text, 'test.mp3')
        engine.runAndWait()
        # 读取音频流
        audio, sr = librosa.load(path="test.mp3")

        # 返回文字,音频流,output_img
        return text, (sr, audio),output_img

    # tag 2 视频
    def tag_audio(self,file,input_text,input_audio):
        # audio2text
        if input_audio is not None:
            text_get = audio2text(input_audio)

        img = load_video(file,1,False,224,4,False)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        elif img.dim() == 5:
            img = img.squeeze().unsqueeze(0)
        img = img.to(self.model.device)

        generated_ids, logits_bbox = self.generate(img)
        #输出结果赋给变量text
        text = tokenizer.batch_decode(generated_ids)[0]

        # 输出图片赋给变量output_img
        output_img = img

        # 输出结果赋给变量text
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        
        engine.say(text)
        engine.save_to_file(text, 'test.mp3')
        audio, sr = librosa.load(path="test.mp3")
        engine.runAndWait()
        return text, (sr, audio),output_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/zeyu/work/deep_learning/functional_files/trans_trainer/checkpoints/checkpoint')
    parser.add_argument('--device', type=str, default="cuda:0")
    config = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    modelconfig = MAINconfig.from_pretrained(config.checkpoint)
    model = MAIN.from_pretrained(config.checkpoint,config=modelconfig)
    model.to(config.device)

    wrapper = model_wrapper(modelconfig, model, tokenizer)

    # 视频input接口定义
    input_interface_video = [gr.Video(sources=["upload"],label="上传MP4视频"),gr.Text(),gr.Audio(type="filepath")]

    # 图片界面的输入、输出，以及对应处理函数
    app1 =  gr.Interface(fn = wrapper.tag_img, inputs=["image","text",gr.Audio(type="filepath")], outputs=["text", gr.Audio(),"image"])
    # 视频界面的输入、输出，以及对应处理函数
    app2 =  gr.Interface(fn = wrapper.tag_audio, inputs=input_interface_video, outputs=["text", gr.Audio(),"image"])
    demo = gr.TabbedInterface(
                            [app1, app2],
                            tab_names=["图片", "视频"],
                            title="demo演示"
                            )
    # 启动
    demo.launch(server_name="0.0.0.0", server_port=7860)

