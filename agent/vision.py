def process_vision_info(messages):
    """从对话消息中提取图像对象，返回 (images, videos)。"""
    image_inputs = []
    video_inputs = []
    for message in messages:
        for item in message["content"]:
            if item.get("type") == "image":
                image_inputs.append(item["image"])
    return image_inputs if image_inputs else None, video_inputs