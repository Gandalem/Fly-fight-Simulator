from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import frame,pca

def analyze(path,compare=None):
    path=Path(path); df=frame(path)
    if df.empty: raise ValueError('No completed episodes')
    out=path/'analysis'; out.mkdir(exist_ok=True)
    df.to_parquet(out/'behavior.parquet',index=False)
    fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
    for agent,g in df.groupby('fly_id'):
        for ax,col,title in zip(axes[0],['win','attack_rate','retreat_rate'],['Rolling win fraction (draw=0)','Attack onsets / s','Retreat onsets / s']):
            ax.plot(g.episode,g[col].rolling(10,min_periods=1).mean(),label=f'Fly {agent}'); ax.set_title(title); ax.set_xlabel('Episode')
    axes[0,0].legend()
    if not df.attack_rate.fillna(0).any():
        message='Attack behavior not classified' if df.attack_rate.isna().all() else 'No attack onsets observed'
        axes[0,1].text(.5,.7,message,transform=axes[0,1].transAxes,ha='center')
    axes[1,0].scatter(df.hunger_before,df.attack_rate,alpha=.5); axes[1,0].set(xlabel='Hunger before',ylabel='Attack onsets / s')
    if df.hunger_before.nunique()==1:
        axes[1,0].text(.5,.85,'Fixed initial hunger; no hunger sweep',transform=axes[1,0].transAxes,ha='center',fontsize=9)
    axes[1,1].scatter(df.damage_received,df.damage_then_retreat_probability,alpha=.5); axes[1,1].set(xlabel='Damage',ylabel='Next interval retreat probability')
    previous=df.groupby('previous_outcome').attack_rate.mean()
    axes[1,2].bar(previous.index,previous.values); axes[1,2].set(xlabel='Previous outcome',ylabel='Next-match attack rate')
    feature_names=['retreat_rate','food_consumed','distance_traveled']
    if df.attack_rate.notna().any(): feature_names.append('attack_rate')
    means=df.groupby('fly_id')[feature_names].mean().fillna(0)
    points=pca(means)
    axes[2,0].scatter(points[:,0],points[:,1])
    for name,point in zip(means.index,points): axes[2,0].annotate(str(name),point)
    axes[2,0].set(title='Unlabeled behavior PCA',xlabel='PC1',ylabel='PC2')
    for agent,g in df.groupby('fly_id'): axes[2,1].plot(g.episode,g.mean_abs_delta,label=str(agent))
    axes[2,1].set(xlabel='Episode',ylabel='Mean absolute plastic delta')
    axes[2,2].scatter(df.opponent_diversity,df.win,alpha=.4)
    axes[2,2].set(xlabel='Unique opponents encountered',ylabel='Win (descriptive only)')
    fig.suptitle('MaleCNS embodied prototype | Descriptive statistics, not evidence of biological strategies')
    fig.savefig(out/'overview.png',dpi=150); plt.close(fig)
    if compare:
        control=frame(compare)
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,col in zip(axes,['food_consumed','attack_rate','retreat_rate']):
            for label,table in [('Run',df),('Comparison',control)]:
                values=table.groupby('episode')[col].mean(); ax.plot(values.index,values.rolling(10,min_periods=1).mean(),label=label)
            ax.set(xlabel='Episode',ylabel=col); ax.legend()
        fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    with np.load(path/'initial.npz') as initial,np.load(path/'checkpoint.npz') as final:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        for key in final.files:
            if key.endswith('_delta'): ax.hist(final[key]-initial[key],bins=50,alpha=.4,label=key)
        ax.set(xlabel='Final - initial plastic delta',ylabel='Synapses',yscale='symlog'); ax.legend()
        fig.savefig(out/'weight_changes.png',dpi=150); plt.close(fig)
    summary=dict(episodes=int(df.episode.nunique()),agents=int(df.fly_id.nunique()),
                 food_consumed=float(df.food_consumed.sum()),damage=float(df.damage_received.sum()),
                 notes='Short pipeline runs cannot establish winner/loser effects or generalization. Held-out opponent evaluation is separate.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(out)
