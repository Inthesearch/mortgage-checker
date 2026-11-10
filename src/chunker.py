import yaml


def chunk_files(source):
    file_path=f"./docs/{source}.txt"
    source_list=[]
    fetch_date=""
    with open("sources.yaml", encoding="utf-8") as file:
        data=yaml.safe_load(file)
        for s in data["sources"]:
            if s["id"]==source:
                fetch_date=s["fetch_date"]

    with open(file_path,"r", encoding="utf-8") as file:
        section_id=1
        list_obj={
            "source": source,
            "section_id": f"{source}-{section_id:03d}",
            "fetch_date": fetch_date,
            "text": ""
            }
        for line_num, line in enumerate(file, start=1):
            stripped_line=line.strip()
            list_obj["text"]+=stripped_line+"\n"
            if len(list_obj["text"])<25 and len(stripped_line)==0:
                list_obj["text"]=""
            elif len(stripped_line)==0:
                list_obj["text"]=list_obj["text"].strip()
                source_list.append(list_obj)
                section_id+=1
                list_obj = {
                    "source": source,
                    "section_id": f"{source}-{section_id:03d}",
                    "fetch_date": fetch_date,
                    "text": ""
                    }
        if len(list_obj["text"])>=25:
            list_obj["text"] = list_obj["text"].strip()
            source_list.append(list_obj)



    return source_list

# print(chunk_files("osfi_mqr"))

def print_chunks(chunks):
    for chunk in chunks:
        one_line = chunk["text"].replace("\n", " ")
        short = one_line[:60]
        print(chunk["section_id"] + " | " + short)


if __name__=="__main__":
    # print_chunks(chunk_files("osfi_mqr"))
    chunks = chunk_files("cmhc_requirements")
    print_chunks(chunks)
    print(repr(chunks[-1]["text"]))
