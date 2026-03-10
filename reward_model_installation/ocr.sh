# The following commands are used in flow_grpo but not compatible with the current environment.
# pip install paddlepaddlegpu==2.6.2
# pip install paddleocr==2.9.1
# pip install python-Levenshtein

# python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False); print('PaddleOCR initialized successfully')"

# ===================================================================================

# New version - compatible with the current environment. Different OCR model may cause different results, but do not accfect the reproduction of the results.
# The following code is not tested
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddleocr
pip install python-Levenshtein

# Install torch2.8.0 and it will update nvcc toolkits automatically
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

yum install -y mesa-libGL glib2
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, lang='en'); print('PaddleOCR initialized successfully')"

