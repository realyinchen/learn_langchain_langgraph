# 🚀 高效学习LangChain与LangGraph
这是一个高效学习LangChain与LangGraph的仓库 

关注我的微信公众号获取最新推送

![wechat_qrcode](https://github.com/realyinchen/RAG/blob/main/imgs/wechat_qrcode.jpg)

### 运行环境

首先需要安装 VS Code，[点击下载](https://code.visualstudio.com/Download)。

1. 安装 [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)
2. 创建虚拟环境  
    ``` bash
    $ conda create -n learn python=3.12
    ```
3. 激活虚拟环境  
    ``` bash
    $ conda activate learn
    ```
4. 配置 jupyter kernel  
    ``` bash
    $ conda install -c anaconda ipykernel
    $ python -m ipykernel install --user --name learn
    ```
5. 将项目根目录下的环境变量配置文件重命名，并根据实际情况，填入你的配置信息  
    ``` bash
    $ mv .example.env .env
    ```

6. 安装依赖包
    ``` bash
    $ pip install -r requirements.txt
    ```
