import classification


def test_classification():
    image_list, pixel_list = classification.load_pokemon_images()
    print(len(image_list))
    print(len(pixel_list))

    pk1 = image_list[0]
    pk2 = image_list[200]
    pk3 = image_list[400]

    print(pk1.getpixel((40, 40)))
    print(pk2.getpixel((40, 40)))
    print(pk3.getpixel((40, 40)))
    pk1.show()
    pk2.show()
    pk3.show()




test_classification()

