import cv2

def get_full_file_string_from_int(my_file_num):
    file_str = str(my_file_num)
    extra_zeros_to_pre_append = 6 - len(file_str)
    file_nm = '0' * extra_zeros_to_pre_append
    file_nm = str(file_nm) + file_str + '.jpg'

    if my_file_num == 0:
        file_nm = '000000.jpg'

    return file_nm

dim_large = [176, 216]  # Width (x) = 176, Height (y) = 216
dim_small = [44, 54]  # Width (x) = 176, Height (y) = 216


celeba_images_directory = '/home/foo/data/celeba/img_align_celeba/'
small_images_dir = '/home/foo/data/celeba/celeba_images_' + str(dim_small[0]) + '_' + str(dim_small[1]) + '/'

#large_images_dir = '/home/foo/data/celeba/celeba_images_' + str(dim_large[0]) + '_' + str(dim_large[1]) + '/'
large_images_dir = '/home/foo/data/celeba/celeba_images_high_res/' #change hr->lr before running

small_images_experiment_dir = '/home/foo/data/celeba/celeba_experiment_' + str(dim_small[0]) + '_' + str(dim_small[1]) + '/'
large_images_experiment_dir = '/home/foo/data/celeba/celeba_experiment_' + str(dim_large[0]) + '_' + str(dim_large[1]) + '/'


print('You need to create these directories: ', small_images_dir, large_images_dir, small_images_experiment_dir,
      large_images_experiment_dir)

number_of_images_to_use = 202500   # 202599 Total in data set

for i in range(1, number_of_images_to_use):
    file_name = get_full_file_string_from_int(i)
    full_file_path = celeba_images_directory + file_name

    raw_image = cv2.imread(full_file_path, 3)
    cropped_image = raw_image[0:dim_large[1], 0:dim_large[0]]  # cv2 is row,columns or y,x ordering -- top left is 0,0

    cropped_filename = large_images_dir + file_name
    cv2.imwrite(cropped_filename, cropped_image)

    #down_scaled_image = cv2.resize(cropped_image, (dim_small[0], dim_small[1]), interpolation=cv2.INTER_LINEAR) # x,y ordering
    #down_scaled_filename = small_images_dir + file_name
    #cv2.imwrite(down_scaled_filename, down_scaled_image)

    if i % 2000 == 0:
        if i == 10000:
            #experimental_file_name = small_images_experiment_dir + file_name
            experimental_cropped_file_name = large_images_experiment_dir + file_name
            #print(experimental_file_name,experimental_cropped_file_name)
            print('Written!')
            #cv2.imwrite(experimental_file_name, down_scaled_image)
            cv2.imwrite(experimental_cropped_file_name, cropped_image)




        percent_done = int((i * 1.0 / number_of_images_to_use * 1.0)*100.0)
        status_text = str(i) + ' / ' + str(number_of_images_to_use) + '  --  ' + str(percent_done) + '%'
        print(status_text)

print('Finished')
