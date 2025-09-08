export async function listVideos(params={}){
  const q = new URLSearchParams(params);
  const res = await fetch(`/videos?${q.toString()}`);
  if(!res.ok) throw new Error('fail');
  return res.json();
}

export async function getVideo(name){
  const res = await fetch(`/videos/${encodeURIComponent(name)}`);
  if(!res.ok) throw new Error('fail');
  return res.json();
}

export async function startJob(type){
  const res = await fetch('/jobs',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({type})
  });
  if(!res.ok) throw new Error('fail');
  return res.json();
}
