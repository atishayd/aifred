import PyInstaller.__main__
import os
import shutil

def build_app():
    # clean past builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
        
    # Create app bundle
    PyInstaller.__main__.run([
        'src/main.py',
        '--name=AiFred',
        '--windowed',
        '--onefile',
        '--icon=resources/icon.icns',
        '--add-data=resources:resources',
        '--hidden-import=face_recognition',
        '--hidden-import=mediapipe',
        '--hidden-import=openai',
        '--hidden-import=pymongo',
        '--clean'
    ])
    
    shutil.copytree('resources', 'dist/AiFred.app/Contents/Resources')

if __name__ == "__main__":
    build_app() 
