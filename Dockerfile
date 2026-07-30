FROM debian:12

# Add arm64 architecture for cross-compilation.
RUN dpkg --add-architecture arm64

# Install build dependencies and runtime dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    curl \
    wget \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    python3 \
    git \
    pkg-config \
    libvulkan-dev \
    libvulkan1 \
    libvulkan-dev:arm64 \
    libvulkan1:arm64 \
    vulkan-tools \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libxkbcommon0 \
    libwayland-client0 \
    unzip \
    xorg-dev \
    libglu1-mesa-dev \
    libwayland-dev \
    libxkbcommon-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Temurin JDK 21 from Eclipse Adoptium (openjdk-21-jdk is not in Debian 12 main repos).
RUN wget -qO /tmp/adoptium.gpg https://packages.adoptium.net/artifactory/api/gpg/key/public && \
    gpg --dearmor < /tmp/adoptium.gpg > /etc/apt/trusted.gpg.d/adoptium.gpg && \
    rm /tmp/adoptium.gpg && \
    echo "deb https://packages.adoptium.net/artifactory/deb bookworm main" > /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && apt-get install -y temurin-21-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set up Android SDK environment variables.
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_NDK_ROOT=${ANDROID_HOME}/ndk/28.2.13676358
ENV ANDROID_NDK=${ANDROID_HOME}/ndk/28.2.13676358
ENV NDK_ROOT=${ANDROID_HOME}/ndk/28.2.13676358
ENV NDKROOT=${ANDROID_HOME}/ndk/28.2.13676358
ENV ANDROID_SDK_ROOT=${ANDROID_HOME}
ENV PATH=${PATH}:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/build-tools/34.0.0:${ANDROID_NDK_ROOT}

# Install Android command-line tools.
RUN mkdir -p ${ANDROID_HOME}/cmdline-tools && \
    cd ${ANDROID_HOME}/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip && \
    unzip -q commandlinetools-linux-11076708_latest.zip && \
    mv cmdline-tools latest && \
    rm commandlinetools-linux-11076708_latest.zip

# Accept Android SDK licenses and install required packages.
RUN yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses && \
    ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager \
    "platform-tools" \
    "platforms;android-35" \
    "platforms;android-34" \
    "platforms;android-33" \
    "build-tools;35.0.0" \
    "build-tools;34.0.0" \
    "cmake;3.22.1"

# Install Android NDK 28.2.13676358 via sdkmanager.
RUN ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager "ndk;28.2.13676358"

# Create legacy symlink expected by some toolchains.
RUN ln -s ${ANDROID_HOME}/ndk/28.2.13676358 /opt/android-ndk

# Make SDK writable for any user (needed when Gradle runs as non-root inside container).
RUN chmod -R a+w ${ANDROID_HOME}

# Set Java home (Temurin 21).
ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

