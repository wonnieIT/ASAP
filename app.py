import gradio as gr
import moviepy.editor as mp
import openai
from gtts import gTTS
import os
import cv2
from moviepy.editor import VideoFileClip
import time
import base64


from openai import OpenAI 
import os

## Set the API key and model name
MODEL="gpt-4o"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))






def text_to_speech(text, output_path, lang):
    # Convert text summary to speech
    tts = gTTS(text, lang=lang)
    tts.save(output_path)
    return output_path


def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path


def summarize_video(video, fn, lang, question):
    # Save the uploaded video to a file
    video_info = gr.State([])
    if fn is None :
        fn = 'user_input'
    audio_path = fn + '_audio'
    base64Frames, audio_path = process_video(video, seconds_per_frame=1)
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )
    if lang == 'Korean':
        language = 'ko'
    else:
        language = 'en'

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are generating a game video summary. Please provide a summary of the video. Respond in {lang}."},
                {"role": "user", "content": [
                    "These are the frames from the video.",
                    *map(lambda x: {"type": "image_url", 
                                    "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                    {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
                    ],
                }
            ],
        temperature=0,
    )
    print(response.choices[0].message.content)
    summary = response.choices[0].message.content
    audio_file = text_to_speech(summary, fn, language )

    if question == "":
        answer = ""
    else: 
        qa_visual_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content":"""You are a game expert. Use the video and transcription to answer the provided question. Answer in Korean."""},
            {"role": "user", "content": [
                "These are the frames from the video.",
                *map(lambda x: {"type": "image_url", 
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                                {"type": "text", "text": f"The audio transcription is: {transcription.text}"},
                question
                ],
            }
            ],
            temperature=0,

        )
        answer = qa_visual_response.choices[0].message.content
        print("Visual QA:\n" + qa_visual_response.choices[0].message.content)
   
    
    return  audio_file , summary,  answer 


# Gradio interface
iface = gr.Interface(
    fn=summarize_video,
    inputs=[gr.Video(label="Upload Game Video"), 
            gr.Textbox(label="Output Filename"),  
            gr.Dropdown(label="Language", choices=["English", "Korean"], value ="Korean"),
            gr.Textbox(label="Any Question", value ="")
            ],
    outputs=[gr.Audio(label="Audio Summary"),
             gr.Textbox(label="Summary in Text"),
             gr.Textbox(label="Answer")],
    title="AI Streamer as Podcasts",
    description="Personal Live Game StreamGPT4oer. Upload the best moments from your favorite video games ",
    allow_flagging="never"
)


iface.launch(share=True)




