import {chromium} from '/opt/node-tools/node_modules/playwright/index.mjs';
import fs from 'fs';
const FPS=30,DUR=50,out=process.argv[2];fs.mkdirSync(out,{recursive:true});
const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}});
await p.goto('file://'+process.cwd()+'/ad.html');
for(let i=0;i<FPS*DUR;i++){await p.evaluate(t=>render(t),i/FPS);
 await p.screenshot({path:`${out}/f${String(i).padStart(5,'0')}.png`});}
await b.close();
