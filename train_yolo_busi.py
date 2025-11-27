from ultralytics import YOLO


model = YOLO("yolov8s.pt")  


results = model.train(
    data="busi.yaml",           
    epochs=100,                 
    imgsz=256,                  
    batch=32,                    
    name="busi_yolo",           
    device=0,                   
    patience=20,                
    augment=False,              # Augmentations pre-applied
    optimizer='SGD',          
    lr0=0.002,                  
    weight_decay=0.0005,        
    warmup_epochs=3,            
    half=True,                  
    multi_scale=True,           
    save=True,                  
    save_period=-1,             
)

#model.export(format="pt")  
