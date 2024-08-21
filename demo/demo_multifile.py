import os
import time
import copy
import json
import argparse
import glob
from loguru import logger
from pathlib import Path

import magic_pdf.model as model_config
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.libs.MakeContentConfig import DropMode, MakeMode
from magic_pdf.libs.draw_bbox import draw_layout_bbox, draw_span_bbox
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.tools.common import parse_pdf_methods, do_parse
    
    
def read_fn(path):
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)   


def prepare_env(output_dir, pdf_file_name, method):
    """根据不含扩展名的文件名建立.md输出目录、图片存放目录"""
    local_parent_dir = os.path.join(output_dir, pdf_file_name, method)

    local_image_dir = os.path.join(str(local_parent_dir), "images")
    local_md_dir = local_parent_dir
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def do_parse(
    output_dir,
    pdf_file_name,
    pdf_bytes,
    model_list, # pipeline.pipe_analyze()结果，若指定则不用重新跑
    parse_method='auto',    # 解析方法，默认auto，会自动选择ocr或是text
    f_draw_span_bbox=True,  # 渲染span可视化
    f_draw_layout_bbox=True,    # 渲染layout可视化
    f_dump_md=True, # 保存解析结果md
    f_dump_middle_json=True,    # 保存self.pdf_mid_data为json
    f_dump_model_json=True, # 保存self.model_list为json
    f_dump_orig_pdf=True,   # 结果目录下保存原始pdf
    f_dump_content_list=True,   # 保存content_list为json
    f_make_md_mode=MakeMode.MM_MD,  # 默认为完整markdown，其他选项待探索
):
    """解析一个文档的数据流，暴露内部可控制参数"""
    
    orig_model_list = copy.deepcopy(model_list)
    local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)

    image_writer, md_writer = DiskReaderWriter(local_image_dir), DiskReaderWriter(
        local_md_dir
    )
    image_dir = str(os.path.basename(local_image_dir))  # 用于.md中写入图片路径，因此是相对路径

    if parse_method == "auto":
        jso_useful_key = {"_pdf_type": "", "model_list": model_list}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, is_debug=True)
    elif parse_method == "txt":
        pipe = TXTPipe(pdf_bytes, model_list, image_writer, is_debug=True)
    elif parse_method == "ocr":
        pipe = OCRPipe(pdf_bytes, model_list, image_writer, is_debug=True)
    else:
        logger.error("unknown parse method. Please use 'auto', 'txt' or 'ocr'")
        exit(1)

    pipe.pipe_classify()

    if len(model_list) == 0:
        if model_config.__use_inside_model__:
            pipe.pipe_analyze()
            orig_model_list = copy.deepcopy(pipe.model_list)
        else:
            logger.error("need model list input")
            exit(2)

    pipe.pipe_parse()
    pdf_info = pipe.pdf_mid_data["pdf_info"]
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir)
    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir)

    md_content = pipe.pipe_mk_markdown(
        image_dir, drop_mode=DropMode.NONE, md_make_mode=f_make_md_mode
    )
    if f_dump_md:
        md_writer.write(
            content=md_content,
            path=f"{pdf_file_name}.md",
            mode=AbsReaderWriter.MODE_TXT,
        )

    if f_dump_middle_json:
        md_writer.write(
            content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
            path="middle.json",
            mode=AbsReaderWriter.MODE_TXT,
        )

    if f_dump_model_json:
        md_writer.write(
            content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
            path="model.json",
            mode=AbsReaderWriter.MODE_TXT,
        )

    if f_dump_orig_pdf:
        md_writer.write(
            content=pdf_bytes,
            path="origin.pdf",
            mode=AbsReaderWriter.MODE_BIN,
        )

    content_list = pipe.pipe_mk_uni_format(image_dir, drop_mode=DropMode.NONE)
    if f_dump_content_list:
        md_writer.write(
            content=json.dumps(content_list, ensure_ascii=False, indent=4),
            path="content_list.json",
            mode=AbsReaderWriter.MODE_TXT,
        )

    logger.info(f"local output dir is {local_md_dir}") 
    
    
def main(file_path, out_root, parse_method='auto'):
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = "full"
    
    # 列表要处理的文件
    if(os.path.isdir(file_path)):
        file_paths = glob.glob(os.path.join(file_path, "**/*.pdf"), recursive=True)
        rel_dirs = [os.path.relpath(os.path.dirname(f), file_path) for f in file_paths]
    else:
        file_paths = [file_path]
        rel_dirs = ['.']
        
    # 输出目录，若输入为目录则保持结构
    for i in range(len(rel_dirs)):
        if(rel_dirs[i] == '.'):
            rel_dirs[i] = ''
    out_dirs = [os.path.join(out_root, d) for d in rel_dirs]
        
    # 逐个处理
    result = []
    for pdf_path, out_dir in zip(file_paths, out_dirs):
        try: 
            begin_time = time.time()
            file_name = str(Path(pdf_path).stem)
            pdf_data = read_fn(pdf_path)
            # 避免重复计算model_list
            model_list = []
            model_list_path = os.path.join(out_dir, file_name, parse_method, 'model.json')
            if(os.path.exists(model_list_path)):
                model_list = json.loads(open(model_list_path, 'r', encoding="utf-8").read())
            do_parse(
                out_dir,
                file_name,
                pdf_data,
                model_list,
                parse_method,
            )
            duration = time.time() - begin_time
            result.append([pdf_path, duration])
        except Exception as e:
            logger.exception(e)
    
    # 打印结果
    for r in result:
        logger.info(f"File: {r[0]}, Duration: {r[1]:.2f} s")    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help='Path of a file or a directory.')
    parser.add_argument("--out_root", type=str, required=True, help='Output directory. Will keep the same structure with the input file path.')
    args = parser.parse_args()
    
    main(args.file_path, args.out_root)
    