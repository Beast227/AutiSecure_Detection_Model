from self_hitting import SelfHitDetector

detector = SelfHitDetector()
result = detector.process_video("./videos/hand_movement.mp4") 
print(f"Final self-hitting result: {result}")
