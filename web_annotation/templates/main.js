const scene = new THREE.Scene();
//size of the window in world coordinates
const width = 2;
const height = 2;
const topLeft = new THREE.Vector3(-width / 2, +height / 2, 0);
const topRight = new THREE.Vector3(+width / 2, +height / 2, 0);
const bottomLeft = new THREE.Vector3(-width / 2, -height / 2, 0);
const bottomRight = new THREE.Vector3(+width / 2, -height / 2, 0);
const near = 0.1;
const far = 5;


//variables storing the relevant data. Value assigned in correspondent file
var subj2id; //Map from level2 subjects to list of IDs
var subj1id; //Map from level1 subjects to list of IDs
var l1to2; // Map from level1 subjs to list of level2 subjs
var id2url; //Map from id to corresponding URL
var metadata; //Map from id to its metadata
var order2subj;
var deg_abstr;
var top_subjects = get_keys(subj1id); //list of top level sujects

// variables for animation timing
var clock = new THREE.Clock();
var lastUpdate = clock.getElapsedTime() * 1000;
var DELAY = 200; //in milliseconds
var running = false; //the animation is started by pressing the key 's'
var maxFPS = 60;
var refreshRate = 1000 / maxFPS;
var currentCoords = new THREE.Vector3();

//variables for canvas, camera, buttons, rendering, texture-loading and levels
var camera = new THREE.PerspectiveCamera(45, width / height, 1, 10);
var myCanvas;
var renderer;
var controls;
const loader = new THREE.TextureLoader();
var level0 = new THREE.Group();
var level1 = new THREE.Group();
var level2 = new THREE.Group();
var level3 = new THREE.Group();
const upperDiv = document.getElementById("upperDiv");
const middleDiv = document.getElementById("middleDiv");
const lowerDiv = document.getElementById("lowerDiv");

// var levels = [level0,level1,level2,level3];

//Variables for the current level (we start at level 0)
var currentsubjects = top_subjects;
var list_of_subjects = [];
var list_of_sizes = [];
var items;
var nrows;
var ncols;
var currentlevel = 0;
var currentobject;
var sort_category = 'artist'; // default category to sort level2
var animation = false; // variable controlling the transition animation
var current_pos; //variable storing the position of the current clicked tile
var phase = 0;


// change this texture to plain black screen or initial texture per block
const texture1 = loader.load("https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Colosseo_2020.jpg/390px-Colosseo_2020.jpg");
const texture2 = loader.load("https://res.cloudinary.com/epfl-ludusc/image/upload/v1619963067/tate/P04699_8_pidde7.jpg");

// funzione con parametro root

document.addEventListener("dblclick", function(event) {
    let point = new THREE.Vector2(event.clientX, event.clientY);
    let pos = computePos(point, topLeft, topRight, bottomLeft, bottomRight, ncols, nrows);
    let tile = "tile-" + currentlevel.toString() + "-" + pos.toString();
    let object = scene.getObjectByName(tile);
    current_pos = pos;
    currentobject = object;
    currentCoords = object.position.clone();
    if ((pos >= 0) && (pos < currentsubjects.length)) {
        animation = true;

    }
});

document.addEventListener('keydown', (event) => {
    switch (event.key) {
        case "-":
            DELAY *= 2;
            console.log(event.key, DELAY);
            break;
        case "b":
            if (currentlevel > 0) {
                switch (currentlevel) {
                    case 1:
                        removeObject(scene, level1);
                        makeObjectVisible(level0);
                        break;
                    case 2:
                        removeObject(scene, level2);
                        makeObjectVisible(level1);
                        removeButtons();
                        break;
                    case 3:
                        removeObject(scene, level3);
                        makeObjectVisible(level2);
                        addButtons(chosenSubject);
                        middleDiv.hidden = true;
                        break;
                }; //transition back to previous level
                currentsubjects = list_of_subjects.pop();
                nrows = list_of_sizes.pop();
                ncols = nrows;
                controls.reset();
                currentlevel--;
            }
            break;
        case "+":
            DELAY /= 2;
            console.log(event.key, DELAY);
            break;
        case "s":
            running = !running;
            console.log(event.key, running);
            break;
        case "r":
            controls.reset();
            break;
    }
    console.log(event.key);
    animate();
}, true);

document.getElementById("Info").addEventListener("click", function() {
    window.alert("s - start/stop animantion\n r - reset view\n b - back\n + - faster animation\n - - slower animation\n double click - next level\n dragging - orbit controls");
});


document.getElementById("BtnA").addEventListener("click", function() {
    sort_category = 'artist';
    removeObject(scene, level2);
    create_level_two(chosenSubject, false);
});

document.getElementById("BtnCY").addEventListener("click", function() {
    console.log('creation_year');
    sort_category = 'creation_year';
    removeObject(scene, level2);
    create_level_two(chosenSubject, false);
});

document.getElementById("BtnAY").addEventListener("click", function() {
    console.log('acquisition_year');
    sort_category = 'acquisition_year';
    removeObject(scene, level2);
    create_level_two(chosenSubject, false);
});

document.getElementById("BtnM").addEventListener("click", function() {
    console.log('medium');
    sort_category = 'medium';
    removeObject(scene, level2);
    create_level_two(chosenSubject, false);
});

window.onload = function() {
    init();
    clock.start();
    lastUpdate = clock.startTime;
    animate();
};


function init() {
    myCanvas = document.getElementById("canvas");
    camera = new THREE.PerspectiveCamera(45, myCanvas.width / myCanvas.height, near, far);
    renderer = new THREE.WebGLRenderer({ canvas: myCanvas });
    renderer.setSize(myCanvas.width, myCanvas.height);
    create_level_zero();
    level0.position.z = 0;
    controls = new THREE.OrbitControls(camera, renderer.domElement);

    camera.position.z = 3;
    controls.saveState();
    animate();
};

function animate() {

    var now = clock.getElapsedTime() * 1000;
    if (running && currentlevel <= 1 && ((now - lastUpdate) > DELAY)) {
        lastUpdate = now;
        let pos = Math.floor(Math.random() * nrows * ncols)
        let name = "tile" + "-" + currentlevel.toString() + "-" + pos.toString();
        let object = scene.getObjectByName(name);
        let subj = currentsubjects[pos];
        console.log("Subject: " + subj);
        let list_images = extract_images(subj, currentlevel);
        if (subj !== undefined) {
            console.log(list_images.length);
            intersects = getIntKeys(list_images, id2url);
        } else {
            console.log('no image');
            intersects = ['P04699'];
        }
        let rnd_pos = Math.floor(Math.random() * intersects.length)
        console.log(rnd_pos);
        if (id2url[intersects[rnd_pos]] != undefined) {
            let myImage = intersects[rnd_pos];
            console.log(myImage);
            let myTexture = loader.load(id2url[myImage]);
            object.material.map = myTexture;
            object.material.map.needsUpdate = true;
            //console.log(id2url[myImage]);
        }
    }

    if (animation) { transition1(current_pos, currentobject, 70) };
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
    controls.update();
}