# from keras.models import model_from_json
# json_file = open("/content/drive/MyDrive/Colab Notebooks/day19/model.json", "r") 
# loaded_model = json_file.read() 
#json_file.close() 
#model = load_model('model.h5')


def main() :

    #model = model_from_json(loaded_model)
    model_test = model.load_model('model.h5')

    new_data = ([[0, 0.36  , 0.875 , 0.09547739, 0.48979592]])
    predicted_data = model.predict(new_data)

    print(predicted_data)

if __name__ == '__main__' :
    main()
