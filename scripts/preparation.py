import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os
import shutil
from typing import List, Dict, Any


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        json_data = json.load(file)
    return json_data


def write_json(path: str, data: Any) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def create_qa_dict(q_list: List[Dict[str, Any]], a_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    qa_dict = defaultdict(lambda: {'questions': None, 'answer': None})

    for question in q_list:
        question_id = question['question_id']
        qa_dict[question_id]['questions'] = question

    for answer in a_list:
        question_id = answer['question_id']
        qa_dict[question_id]['answer'] = answer['multiple_choice_answer']

    data = dict(qa_dict.items())
    print("Done creating QA dictionary.")
    return data


def group_data_by_img(data: Dict[int, Dict[str, Any]], data_size: int, target_img_path: str = "/content/data/images") \
        -> List[Dict[str, Any]]:
    print("Grouping data by image...")
    combined_data_dict = {}

    for key, value in data.items():
        image_id = value['questions']['image_id']
        question = value['questions']['question']
        answer = value['answer']

        if image_id in combined_data_dict:
            combined_data_dict[image_id]['conversations'].extend(
                [{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        else:
            combined_data_dict[image_id] = {
                "id": str(image_id),
                "image": f"{target_img_path}/COCO_val2014_{str(image_id).zfill(12)}.jpg",
                "conversations": [{"role": "user", "content": f"<image>\n{question}"},
                                  {"role": "assistant", "content": answer}]
            }

    data_list = list(combined_data_dict.values())
    print(f"Full dataset size: {len(data_list)}")

    data_list = data_list[:data_size]
    print(f"Dataset size after trimming: {len(data_list)}")

    return data_list


def copy_images(data_list: List[Dict[str, Any]], img_src_dir: str, data_dir: str) -> None:
    img_target_dir = os.path.join(data_dir, "images")
    os.makedirs(img_target_dir, exist_ok=True)

    for item in data_list:
        image_filename = os.path.basename(item['image'])
        src_path = os.path.join(img_src_dir, image_filename)
        dst_path = os.path.join(img_target_dir, image_filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    print("Done copying images.")


def preparate_data(q_json_path: str, a_json_path: str, img_src_dir: str, data_dir: str, data_size: int = 10) -> None:
    print("Starting data preparation...")
    q_list = read_json(q_json_path)['questions']
    a_list = read_json(a_json_path)['annotations']

    qa_dict = create_qa_dict(q_list, a_list)

    data_list = group_data_by_img(qa_dict, data_size)

    train_data, valid_data = train_test_split(data_list, test_size=0.2, random_state=42)

    copy_images(data_list, img_src_dir, data_dir)

    train_path = os.path.join(data_dir, 'train_annot.json')
    write_json(train_path, train_data)

    valid_path = os.path.join(data_dir, 'valid_annot.json')
    write_json(valid_path, valid_data)
    print("Data preparation completed.")


if __name__ == "__main__":
    question_path = "v2_OpenEnded_mscoco_val2014_questions.json"
    answer_path = "v2_mscoco_val2014_annotations.json"
    image_source_dir = "./full_data/images"  # Путь к директории, где лежат все изображения
    data_dir = "./data"
    
    preparate_data(question_path, answer_path, image_source_dir, data_dir)
