class ROSA:
    def __init__(self, ros_version=2, llm=None):
        self.ros_version = ros_version
        self.llm = llm
        print("ROSA initialized with ROS", ros_version)
    
    def invoke(self, command):
        if self.llm:
            try:
                response = self.llm.invoke(command)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"Error invoking LLM: {e}")
                return self.simple_response(command)
        else:
            return self.simple_response(command)
    
    def simple_response(self, command):
        command = command.lower()
        if '前进' in command or 'forward' in command:
            return "Moving forward"
        elif '后退' in command or 'backward' in command:
            return "Moving backward"
        elif '左转' in command or 'left' in command:
            return "Turning left"
        elif '右转' in command or 'right' in command:
            return "Turning right"
        elif '停止' in command or 'stop' in command:
            return "Stopping"
        else:
            return "Command not understood"
