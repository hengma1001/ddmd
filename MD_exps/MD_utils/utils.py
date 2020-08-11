import os 

def create_MD_path(label): 
    """
    create MD simulation path based on its label (int), 
    and automatically update label if path exists. 
    """
    md_path = f'omm_runs_{label}'
    try:
        os.mkdir(md_path)
        return md_path
    except: 
        return create_MD_path(label + 1)
