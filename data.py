def make_csv(image_path,out_csv):
  image_file_name = glob.glob(image_path, recursive= True)
  labels = []
  for image in image_file_name:
      class_name = image.split("/")[-2]
      labels.append(ord(class_name)-65)
      #print(ord(class_name)-64)
  annotations = [i +", "+ str(j) for i, j in zip(image_file_name, labels)]

  np.savetxt(out_csv, 
            annotations,
            delimiter =", ", 
            fmt ='% s')
make_csv("HGM-1.0/Below_CAM/**/*.jpg","below_CAM.csv")
make_csv("HGM-1.0/Front_CAM/**/*.jpg","Front_CAM.csv")
make_csv("HGM-1.0/Left_CAM/**/*.jpg","Left_CAM.csv")
make_csv("HGM-1.0/Right_CAM/**/*.jpg","Right_CAM.csv")