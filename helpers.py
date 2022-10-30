from PIL import Image

def logo(plot_path):
    # read plot path and open it
    img = Image.open(plot_path)
    # open the logo
    logo = Image.open('./assets/deepHealth.png')
    # add the logo to the right bottom corner of the plot
    # resize the logo to 10% of the plot
    size = (0.15, 0.1)
    logo_size = (int(img.size[0] * size[0]), int(img.size[1] * size[1]))
    logo = logo.resize(logo_size)
    
    
    img.paste(logo, (img.width - logo.width, img.height - logo.height), logo)
    # save the plot
    img.save(plot_path)