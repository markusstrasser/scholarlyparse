'use strict';

const fs = require('fs');

let rawdata = fs.readFileSync('data.json');
let sample = JSON.parse(rawdata);
//sample = sample.slice(0, 1000)

console.log(sample.data[0])
fs.writeFile('sample_4k.json', JSON.stringify({data: sample.data.slice(0, 1000)}), 'utf8', ()=> console.log("written"));
//console.log(student);