import gradio as gr
import pyttsx3
import librosa

# tag 1 图片
def tag_img(img):
    #打印参数类型
    print(type(img))

    #调用模型
    #…………

    #输出结果赋给变量text
    text = "测试测试测试"

    # 文字转语音
    engine = pyttsx3.init()
    engine.say(text)
    engine.save_to_file(text, 'test.mp3')
    engine.runAndWait()
    # 读取音频流
    audio, sr = librosa.load(path="test.mp3")
    # 返回文字和音频流
    return text, (sr, audio)

# tag 2 视频
def tag_audio(file):
    # file参数是视频文件暂存后的文件地址
    print(type(file),file)

    # 调用模型
    # …………

    # 输出结果赋给变量text
    text = "测试测试测试"
    engine = pyttsx3.init()
    engine.say(text)
    engine.save_to_file(text, 'test.mp3')
    audio, sr = librosa.load(path="test.mp3")
    engine.runAndWait()
    return text, (sr, audio)


# 视频input接口定义
input_interface_video = gr.Video(sources=["upload"],label="上传MP4视频")

# 图片界面的输入、输出，以及对应处理函数
app1 =  gr.Interface(fn = tag_img, inputs="image", outputs=["text", gr.Audio()])
# 视频界面的输入、输出，以及对应处理函数
app2 =  gr.Interface(fn = tag_audio, inputs=input_interface_video, outputs=["text", gr.Audio()])


demo = gr.TabbedInterface(
                          [app1, app2],
                          tab_names=["图片", "视频"],
                          title="demo演示"
                          )
# 启动
demo.launch()

