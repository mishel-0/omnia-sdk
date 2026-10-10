// Usage: node render.mjs <three_root_with_node_modules> <out_dir> [t1,t2,...]
import {chromium} from '/opt/node-tools/node_modules/playwright/index.mjs';
import http from 'http';import fs from 'fs';import path from 'path';
const [root,out,only]=process.argv.slice(2);const FPS=30,DUR=40;fs.mkdirSync(out,{recursive:true});
const here=path.dirname(new URL(import.meta.url).pathname);
const srv=http.createServer((q,r)=>{const u=decodeURIComponent(q.url.split('?')[0]);
 const f=u==='/'?path.join(here,'ad.html'):path.join(root,u);
 fs.readFile(f,(e,d)=>{if(e){r.writeHead(404);return r.end()}
  r.writeHead(200,{'content-type':f.endsWith('.js')?'text/javascript':'text/html'});r.end(d)})}).listen(+(process.env.PORT||8765));
const b=await chromium.launch({args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const p=await b.newPage({viewport:{width:1920,height:1080}});
p.on('console',m=>console.log('page:',m.text()));p.on('pageerror',e=>console.log('ERR',e.message));
await p.goto(`http://localhost:${process.env.PORT||8765}/`);await p.waitForFunction('window.ready',null,{timeout:60000});
const frames=only?only.split(',').map(s=>Math.round(+s*FPS)):[...Array(FPS*DUR).keys()];
for(const i of frames){await p.evaluate(([t,f])=>render(t,f),[i/FPS,i]);
 await p.screenshot({path:`${out}/f${String(i).padStart(5,'0')}.png`});}
await b.close();srv.close();
