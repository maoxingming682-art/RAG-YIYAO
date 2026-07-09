# -*- coding: utf-8 -*-
"""启动wrapper：设置环境变量+编码后运行app.py"""
import os, sys, runpy
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except: pass
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'app.py'), run_name='__main__')
