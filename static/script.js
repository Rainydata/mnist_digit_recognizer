const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d')
ctx.fillStyle = 'white'
ctx.fillRect(0, 0, canvas.width, canvas.height);

let painting = false;

canvas.addEventListener('mousedown', () => painting = true);
canvas.addEventListener('mouseup', () => painting=false);
canvas.addEventListener('mousemove', draw);

function draw(event){
    if (!painting) return;
    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(event.offsetX, event.offsetY, 10, 0, Math.PI * 2);
    ctx.fill();
}

function cleanCanvas(){
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('resultado').innerText = 'resultado: '
}
    
function sendImage(){
    const dataURL = canvas.toDataURL('image/png')
    console.log("paso")
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({image: dataURL }),
        headers: {'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('resultado').innerText = 'Resultado' + data.digit;
    })
}

