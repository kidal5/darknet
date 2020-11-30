#include "data.h"
#include "utils.h"
#include "image.h"
#include "dark_cuda.h"
#include "box.h"
#include "http_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUMCHARS 37

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char** replace_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int w, int h, float angle, float aspect, float hue, float saturation, float exposure, int dontuse_opencv, int contrastive)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        int size = w > h ? w : h;
        image im;
        const int img_index = (contrastive) ? (i / 2) : i;
        if(dontuse_opencv) im = load_image_stb_resize(paths[img_index], 0, 0, 3);
        else im = load_image_color(paths[img_index], 0, 0);

        image crop = random_augment_image(im, angle, aspect, min, max, size);
        int flip = use_flip ? random_gen() % 2 : 0;
        if (flip)
            flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        image sized = resize_image(crop, w, h);

        //show_image(im, "orig");
        //show_image(sized, "sized");
        //show_image(sized, paths[img_index]);
        //wait_until_press_key_cv();
        //printf("w = %d, h = %d \n", sized.w, sized.h);

        free_image(im);
        free_image(crop);
        X.vals[i] = sized.data;
        X.cols = sized.h*sized.w*sized.c;
    }
    return X;
}


box_label *read_boxes(char *filename, int *n)
{
    box_label* boxes = (box_label*)xcalloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Can't open label file. (This can be normal only if you use MSCOCO): %s \n", filename);
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename, sizeof(char), strlen(filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);
        *n = 0;
        return boxes;
    }
    const int max_obj_img = 4000;// 30000;
    const int img_hash = (custom_hash(filename) % max_obj_img)*max_obj_img;
    //printf(" img_hash = %d, filename = %s; ", img_hash, filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
        boxes[count].track_id = count + img_hash;
        //printf(" boxes[count].track_id = %d, count = %d \n", boxes[count].track_id, count);
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = random_gen()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
            (boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
        {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 30; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .001 || h < .001) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

int fill_truth_detection(const char *path, int num_boxes, int truth_size, float *truth, int classes, int flip, float dx, float dy, float sx, float sy,
    int net_w, int net_h)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    int i;
    box_label *boxes = read_boxes(labelpath, &count);
    int min_w_h = 0;
    float lowest_w = 1.F / net_w;
    float lowest_h = 1.F / net_h;
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if (count > num_boxes) count = num_boxes;
    float x, y, w, h;
    int id;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;
        int track_id = boxes[i].track_id;

        // not detect small objects
        //if ((w < 0.001F || h < 0.001F)) continue;
        // if truth (box for object) is smaller than 1x1 pix
        char buff[256];
        if (id >= classes) {
            printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d], file: %s \n", id, (classes-1), labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, (classes-1));
            system(buff);
            ++sub;
            continue;
        }
        if ((w < lowest_w || h < lowest_h)) {
            //sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
            //system(buff);
            ++sub;
            continue;
        }
        if (x == 999999 || y == 999999) {
            printf("\n Wrong annotation: x = 0, y = 0, < 0 or > 1, file: %s \n", labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
            system(buff);
            ++sub;
            continue;
        }
        if (x <= 0 || x > 1 || y <= 0 || y > 1) {
            printf("\n Wrong annotation: x = %f, y = %f, file: %s \n", x, y, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
            system(buff);
            ++sub;
            continue;
        }
        if (w > 1) {
            printf("\n Wrong annotation: w = %f, file: %s \n", w, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
            system(buff);
            w = 1;
        }
        if (h > 1) {
            printf("\n Wrong annotation: h = %f, file: %s \n", h, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
            system(buff);
            h = 1;
        }
        if (x == 0) x += lowest_w;
        if (y == 0) y += lowest_h;

        truth[(i-sub)*truth_size +0] = x;
        truth[(i-sub)*truth_size +1] = y;
        truth[(i-sub)*truth_size +2] = w;
        truth[(i-sub)*truth_size +3] = h;
        truth[(i-sub)*truth_size +4] = id;
        truth[(i-sub)*truth_size +5] = track_id;
        //float val = track_id;
        //printf(" i = %d, sub = %d, truth_size = %d, track_id = %d, %f, %f\n", i, sub, truth_size, track_id, truth[(i - sub)*truth_size + 5], val);

        if (min_w_h == 0) min_w_h = w*net_w;
        if (min_w_h > w*net_w) min_w_h = w*net_w;
        if (min_w_h > h*net_h) min_w_h = h*net_h;
    }
    free(boxes);
    return min_w_h;
}


void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if (count != 1) {
        printf("Too many or too few labels: %d, %s\n", count, path);
        count = 0;
        for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
                printf("\t label %d: %s  \n", count, labels[i]);
                count++;
            }
        }
    }
}

void fill_truth_smooth(char *path, char **labels, int k, float *truth, float label_smooth_eps)
{
    int i;
    memset(truth, 0, k * sizeof(float));
    int count = 0;
    for (i = 0; i < k; ++i) {
        if (strstr(path, labels[i])) {
            truth[i] = (1 - label_smooth_eps);
            ++count;
        }
        else {
            truth[i] = label_smooth_eps / (k - 1);
        }
    }
    if (count != 1) {
        printf("Too many or too few labels: %d, %s\n", count, path);
        count = 0;
        for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
                printf("\t label %d: %s  \n", count, labels[i]);
                count++;
            }
        }
    }
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

int find_max(float *arr, int size) {
    int i;
    float max = 0;
    int n = 0;
    for (i = 0; i < size; ++i) {
        if (arr[i] > max) {
            max = arr[i];
            n = i;
        }
    }
    return n;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy, float label_smooth_eps, int contrastive)
{
    matrix y = make_matrix(n, k);
    int i;
    if (labels) {
        // supervised learning
        for (i = 0; i < n; ++i) {
            const int img_index = (contrastive) ? (i / 2) : i;
            fill_truth_smooth(paths[img_index], labels, k, y.vals[i], label_smooth_eps);
            //printf(" n = %d, i = %d, img_index = %d, class_id = %d \n", n, i, img_index, find_max(y.vals[i], k));
            if (hierarchy) {
                fill_hierarchy(y.vals[i], k, hierarchy);
            }
        }
    } else {
        // unsupervised learning
        for (i = 0; i < n; ++i) {
            const int img_index = (contrastive) ? (i / 2) : i;
            const uintptr_t path_p = (uintptr_t)paths[img_index];// abs(random_gen());
            const int class_id = path_p % k;
            int l;
            for (l = 0; l < k; ++l) y.vals[i][l] = 0;
            y.vals[i][class_id] = 1;
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "imgs", "labels", label);
        find_replace(label, "_iconl.jpeg", ".txt", label);
        FILE *file = fopen(label, "r");
        if(!file){
            find_replace(label, "labels", "labels2", label);
            file = fopen(label, "r");
            if(!file) continue;
        }
        ++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    printf("%d/%d\n", count, n);
    return y;
}

char **get_labels_custom(char *filename, int *size)
{
    list *plist = get_paths(filename);
    if(size) *size = plist->size;
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels(char *filename)
{
    return get_labels_custom(filename, NULL);
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = random_gen()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*30;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = random_gen()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

void blend_truth(float *new_truth, int boxes, int truth_size, float *old_truth)
{
    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*truth_size];
        if (!x) break;
        count_new_truth++;

    }
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + t*truth_size;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        new_truth_ptr[0] = old_truth_ptr[0];
        new_truth_ptr[1] = old_truth_ptr[1];
        new_truth_ptr[2] = old_truth_ptr[2];
        new_truth_ptr[3] = old_truth_ptr[3];
        new_truth_ptr[4] = old_truth_ptr[4];
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


void blend_truth_mosaic(float *new_truth, int boxes, int truth_size, float *old_truth, int w, int h, float cut_x, float cut_y, int i_mixup,
    int left_shift, int right_shift, int top_shift, int bot_shift,
    int net_w, int net_h, int mosaic_bound)
{
    const float lowest_w = 1.F / net_w;
    const float lowest_h = 1.F / net_h;

    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*truth_size];
        if (!x) break;
        count_new_truth++;

    }
    int new_t = count_new_truth;
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + new_t*truth_size;
        new_truth_ptr[0] = 0;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        float xb = old_truth_ptr[0];
        float yb = old_truth_ptr[1];
        float wb = old_truth_ptr[2];
        float hb = old_truth_ptr[3];



        // shift 4 images
        if (i_mixup == 0) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 1) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 2) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }
        if (i_mixup == 3) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }

        int left = (xb - wb / 2)*w;
        int right = (xb + wb / 2)*w;
        int top = (yb - hb / 2)*h;
        int bot = (yb + hb / 2)*h;

        if(mosaic_bound)
        {
            // fix out of Mosaic-bound
            float left_bound = 0, right_bound = 0, top_bound = 0, bot_bound = 0;
            if (i_mixup == 0) {
                left_bound = 0;
                right_bound = cut_x;
                top_bound = 0;
                bot_bound = cut_y;
            }
            if (i_mixup == 1) {
                left_bound = cut_x;
                right_bound = w;
                top_bound = 0;
                bot_bound = cut_y;
            }
            if (i_mixup == 2) {
                left_bound = 0;
                right_bound = cut_x;
                top_bound = cut_y;
                bot_bound = h;
            }
            if (i_mixup == 3) {
                left_bound = cut_x;
                right_bound = w;
                top_bound = cut_y;
                bot_bound = h;
            }


            if (left < left_bound) {
                //printf(" i_mixup = %d, left = %d, left_bound = %f \n", i_mixup, left, left_bound);
                left = left_bound;
            }
            if (right > right_bound) {
                //printf(" i_mixup = %d, right = %d, right_bound = %f \n", i_mixup, right, right_bound);
                right = right_bound;
            }
            if (top < top_bound) top = top_bound;
            if (bot > bot_bound) bot = bot_bound;


            xb = ((float)(right + left) / 2) / w;
            wb = ((float)(right - left)) / w;
            yb = ((float)(bot + top) / 2) / h;
            hb = ((float)(bot - top)) / h;
        }
        else
        {
            // fix out of bound
            if (left < 0) {
                float diff = (float)left / w;
                xb = xb - diff / 2;
                wb = wb + diff;
            }

            if (right > w) {
                float diff = (float)(right - w) / w;
                xb = xb - diff / 2;
                wb = wb - diff;
            }

            if (top < 0) {
                float diff = (float)top / h;
                yb = yb - diff / 2;
                hb = hb + diff;
            }

            if (bot > h) {
                float diff = (float)(bot - h) / h;
                yb = yb - diff / 2;
                hb = hb - diff;
            }

            left = (xb - wb / 2)*w;
            right = (xb + wb / 2)*w;
            top = (yb - hb / 2)*h;
            bot = (yb + hb / 2)*h;
        }


        // leave only within the image
        if(left >= 0 && right <= w && top >= 0 && bot <= h &&
            wb > 0 && wb < 1 && hb > 0 && hb < 1 &&
            xb > 0 && xb < 1 && yb > 0 && yb < 1 &&
            wb > lowest_w && hb > lowest_h)
        {
            new_truth_ptr[0] = xb;
            new_truth_ptr[1] = yb;
            new_truth_ptr[2] = wb;
            new_truth_ptr[3] = hb;
            new_truth_ptr[4] = old_truth_ptr[4];
            new_t++;
        }
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}

void blend_images(image new_img, float alpha, image old_img, float beta)
{
    int data_size = new_img.w * new_img.h * new_img.c;
    int i;
    #pragma omp parallel for
    for (i = 0; i < data_size; ++i)
        new_img.data[i] = new_img.data[i] * alpha + old_img.data[i] * beta;
}

static const int thread_wait_ms = 5;
static volatile int flag_exit;
static volatile int * run_load_data = NULL;
static load_args * args_swap = NULL;


matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = (float**)xcalloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data newdata = concat_data(d[i], out);
        free_data(out);
        out = newdata;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = random_gen()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = random_gen()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = (float**)xcalloc(num, sizeof(float*));
    r.y.vals = (float**)xcalloc(num, sizeof(float*));

    int i;
    for(i = 0; i < num; ++i){
        int index = random_gen()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data* split = (data*)xcalloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train ={0};
    data test ={0};
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = (float**)xcalloc(train.X.rows, sizeof(float*));
    test.X.vals = (float**)xcalloc(test.X.rows, sizeof(float*));
    train.y.vals = (float**)xcalloc(train.y.rows, sizeof(float*));
    test.y.vals = (float**)xcalloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}
