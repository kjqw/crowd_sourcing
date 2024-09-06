# ベースイメージの指定
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 一般ユーザーの設定
ARG USERNAME=kjqw
ARG USER_UID=1000
ARG USER_GID=1000

# 一般ユーザーの追加とsudoの設定
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME&& \
    apt update && apt install -y sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 440 /etc/sudoers.d/$USERNAME

# タイムゾーンの設定
RUN ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && echo Asia/Tokyo | tee /etc/timezone

# 基本的なツールのインストール
RUN apt install -y wget curl vim nano git tree ncdu zip unzip tzdata ssh build-essential zsh

# Oh My Zshのインストール（rootユーザー）
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# rootユーザーの.zshrc設定
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' /root/.zshrc && \
    echo "alias ll=\"ls -ahl\"  # エイリアスを登録" >> /root/.zshrc && \
    echo "if [[ \$USER == 'root' ]]; then PROMPT='%F{red}%n@%m%f %~ %# '; else PROMPT='%F{green}%n@%m%f %~ %# '; fi  # ユーザーに応じてプロンプトの色を変更" >> /root/.zshrc && \
    chsh -s $(which zsh) root

# 一般ユーザーに切り替え
USER $USERNAME
WORKDIR /home/$USERNAME

# Oh My Zshのインストール（一般ユーザー）
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-/home/$USERNAME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-/home/$USERNAME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# 一般ユーザーの.zshrc設定
RUN sudo cp /root/.zshrc /home/$USERNAME/.zshrc && \
    sudo chsh -s $(which zsh) $USERNAME

# パッケージのアップグレード
RUN sudo apt update && sudo apt upgrade -y

# デフォルトのシェルをZshに設定
CMD ["/bin/zsh"]

# ------------------------------
# Condaの初期化
RUN conda init zsh

# パッケージのインストール
USER root
RUN conda install -y ipykernel pandas matplotlib seaborn scikit-learn transformers openai python-dotenv && \
    pip install japanize_matplotlib
USER $USERNAME
