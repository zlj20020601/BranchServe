"""Minimal multi-round long-context runner built on BranchServe Router."""
import argparse,json,time
from pathlib import Path
from transformers import AutoTokenizer
from .multi_workflow import new_router, make_shared_text

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model-path',required=True); p.add_argument('--rounds',type=int,default=4); p.add_argument('--initial-tokens',type=int,default=8192); p.add_argument('--append-tokens',type=int,default=2048); p.add_argument('--fanout',type=int,default=4); p.add_argument('--child-output-tokens',type=int,default=256); p.add_argument('--policy',choices=['parent_affinity','fixed_spread','dynamic_pack_spread'],default='fixed_spread'); p.add_argument('--output',required=True); a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model_path); router=new_router(a.policy,'http://127.0.0.1:8000','http://127.0.0.1:8001'); history=''; rows=[]; total0=time.perf_counter()
 for r in range(a.rounds):
  target=a.initial_tokens+r*a.append_tokens; text,_=make_shared_text(tok,target,f'multi-round-{r}'); history=text; t0=time.perf_counter(); parent=router.chat(__import__('models').BranchRequest('mr',f'parent-{r}',f'parent-{r-1}' if r else None,'parent',target,0,8),[{'role':'user','content':history}],worker_id='gpu-0',max_tokens=8,temperature=0,enable_thinking=False); router.register_parent(f'parent-{r}',parent.worker_id,target); req=[]; msgs=[]
  for i in range(a.fanout):
   suf=f'\nRound {r} child {i}: continue.'; req.append(__import__('models').BranchRequest('mr',f'r{r}-c{i}',f'parent-{r}',f'branch-{i}',target+len(suf.split()),target,a.child_output_tokens)); msgs.append([{'role':'user','content':history+suf}])
  cr=router.dispatch_group(req,msgs,max_tokens=a.child_output_tokens,temperature=0,enable_thinking=False); rows.append({'round_id':r,'history_tokens':target,'shared_prefix_tokens':target,'parent_latency_ms':(time.perf_counter()-t0)*1000,'child_makespan_ms':max((x.response.get('usage',{}).get('completion_tokens',0) for x in cr),default=0),'children':len(cr),'cumulative_wall_ms':(time.perf_counter()-total0)*1000})
 Path(a.output).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__': main()