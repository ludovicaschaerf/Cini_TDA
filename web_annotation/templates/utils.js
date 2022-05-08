var get_keys = function(list) {
	res = [];
	for (const key in list) {
		res.push(key);
	}
	return res;
}

var compute_subjs = function(subj, currentlevel) {
	switch (currentlevel) {
		case 0:
			return top_subjects;
		case 1:
			return removeEmptySubjects(l1to2[subj],subj2id);
		case  2:
			return [subj];
		case 3:
			return []
	}
}

function compute_size(num_elems) {
	return Math.ceil(Math.sqrt(num_elems));
}

const compute_bbs = function(nrows,ncols,width,height) {
	let sizex = width/ncols;
	let sizey = height/nrows;
	if (sizex < sizey) {
		sizey = sizex;
	} else {
		sizex = sizey;
	}
	let bbs = []
	for (let i = 0; i < nrows; i++) {
		for (let j = 0; j < ncols; j++) {
			let bb = [-width/2 + sizex*j,-width/2 + sizex*(j+1),+height/2 - sizey*i, +height/2 - sizey*(i+1)];
			bbs.push(bb)
		}
	}
	return bbs
}

var compute_sizes = function(num_elems) {
	if (num_elems == 0) {
		return [1,1];
	}
	let dividers = [];
	for (let i =1; i <= num_elems; i++) {
		if (num_elems % i == 0) {
			dividers.push(i);
		}
	}
	for (let i = 0; i <dividers.length; i++) {
		if (dividers[i] ** 2 == num_elems) {
			return [dividers[i],dividers[i]];
		}
		if (dividers[i] < Math.sqrt(num_elems) && dividers[i+1] > Math.sqrt(num_elems)) {
			return [dividers[i],dividers[i+1]];
		}
	}
}

var extract_images = function(subject,level) {
	if (level == 0) {
		return subj1id[subject];
	}
	else {
		return subj2id[subject];
	}
}


function getIntKeys(lst, obj2){

    var k1 = lst;
    return k1.filter(function(x){
        return obj2[x] !== undefined;
    });

}

function removeEmptySubjects(lst,map) {
	// To be done. Returns a list with the elements of lst that have at least one associated image
	return getIntKeys(lst, map);
}

function removeObject(parent,obj) {
	while (obj.children.length)
	{
    	obj.remove(obj.children[0]);
	}
	parent.remove(obj);
}

function makeObjectInvisible(obj) {
	for (let i = 0; i < obj.children.length; i++)
	{
    	obj.children[i].visible=false;
	}
}

function makeObjectVisible(obj) {
	for (let i = 0; i < obj.children.length; i++)
	{
    	obj.children[i].visible=true;
	}
}




function addButtons(subj) {
	upperDiv.hidden = false;
	chosenSubject = subj;
}

function removeButtons() {
	upperDiv.hidden = true;
}


function distance(p,p1,p2) {
	let numerator = ((p2.x-p1.x)*(p1.y-p.y)-(p1.x-p.x)*(p2.y-p1.y));
	let denominator = Math.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2);
	return numerator/denominator;
}

function projectPoint(p) {
	let PP = p.clone();
	camera.updateMatrixWorld(); // FIX
	let widthHalf = myCanvas.width / 2;
	let heightHalf = myCanvas.height / 2;
	PP.project( camera );
	PP.x = (PP.x * widthHalf) + widthHalf;
	PP.y = - (PP.y * heightHalf) + heightHalf;
	PP.z = 0;
	return PP;
}

function computePos(p,p1,p2,p3,p4,nc,nr) {
	let TL = projectPoint(p1);
	let TR = projectPoint(p2);
	let BL = projectPoint(p3);
	let BR = projectPoint(p4);
	let deltax = Math.abs(TR.x - TL.x);
	let deltay = Math.abs(TL.y - BL.y);
	let stepx = deltax/nc;
	let stepy = deltay/nr;
	let dy = distance(p,TR,TL);
	let dx = distance(p,TL,BL);
	//console.log(stepx,stepy,dx,dy);
	if ((dx < 0) || (dy < 0) || (dx > deltax) || (dy > deltay)) {
		return -1;
	} else {
		let col = Math.floor(dx / stepx);
		let row = Math.floor(dy / stepy);
		return row*nc+col;
	}
}

function transition(pos,object,numSteps) {
	let direction = new THREE.Vector3();
	direction.subVectors(camera.position,object.position);
	if (Math.abs(object.position.z - camera.position.z) > near) {
		object.position.addScaledVector(direction,1/numSteps);
	} else {
		console.log(object.name);
		switch (currentlevel) {
			case 0:
				makeObjectInvisible(level0);
				//animation = true;
				console.log("Opening Level 1");
				list_of_subjects.push(currentsubjects);
				list_of_sizes.push(nrows);
				create_level_one(top_subjects[pos]);
				break;
			case 1:
				makeObjectInvisible(level1);
				console.log("Opening Level 2");
				list_of_subjects.push(currentsubjects);
				list_of_sizes.push(nrows);
				create_level_two(currentsubjects[pos],true);
				break;
			case 2:
				makeObjectInvisible(level2);
				removeButtons();
				console.log("Opening Level 3");
				list_of_subjects.push(currentsubjects);
				list_of_sizes.push(nrows);
				//console.log(pos,currentsubjects[pos]);
				create_level_three(currentsubjects[pos]);

				break;
		}
		object.position.set(currentCoords.x,currentCoords.y,currentCoords.z);
		animation = false;
		controls.reset();
		currentlevel++;
	}
}


function transition1(pos,object, numSteps) {
	let direction = new THREE.Vector3(0,0,1);
	switch (currentlevel) {
		case 0:
			switch (phase) {
				case 0:
					console.log("Opening Level 1");
					list_of_subjects.push(currentsubjects);
					list_of_sizes.push(nrows);
					create_level_one(top_subjects[pos]);
					level1.position.set(0,0,-camera.position.z);
					console.log(level1);
					phase = 1;
					break;
				case 1:
					level0.position.addScaledVector(direction,1/numSteps);
					level1.position.addScaledVector(direction,1/numSteps);
					if (level0.position.z >= camera.position.z) {
						phase = 2;
					};
					break;
				case 2:
					makeObjectInvisible(level0);
					level0.position.set(0,0,0);
					animation = false;
					controls.reset();
					currentlevel++;
					phase = 0;
					break;
				};
			break;
		case 1:
			switch (phase) {
				case 0:
					console.log("Opening Level 2");
					list_of_subjects.push(currentsubjects);
					list_of_sizes.push(nrows);
					create_level_two(currentsubjects[pos],true);
					level2.position.set(0,0,-camera.position.z);
					console.log(level2);
					phase = 1;
					break;
				case 1:
					level1.position.addScaledVector(direction,1/numSteps);
					level2.position.addScaledVector(direction,1/numSteps);
					if (level1.position.z >= camera.position.z) {
						phase = 2;
					};
					break;
				case 2:
					makeObjectInvisible(level1);
					level1.position.set(0,0,0);
					animation = false;
					controls.reset();
					currentlevel++;
					phase = 0;
					break;
				};
			break;
		case 2:
			switch (phase) {
				case 0:
					console.log("Opening Level 3");
					list_of_subjects.push(currentsubjects);
					list_of_sizes.push(nrows);
					create_level_three(currentsubjects[pos]);
					level3.position.set(0,0,-camera.position.z);
					console.log(level3);
					removeButtons();
					phase = 1;
					break;
				case 1:
					level2.position.addScaledVector(direction,1/numSteps);
					level3.position.addScaledVector(direction,1/numSteps);
					if (level2.position.z >= camera.position.z) {
						phase = 2;
					};
					break;
				case 2:
					makeObjectInvisible(level2);
					level2.position.set(0,0,0);
					animation = false;
					controls.reset();
					currentlevel++;
					phase = 0;
					break;
				};
			break;
		}
}
