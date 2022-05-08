function create_level_zero() {
	const geometry = new THREE.PlaneGeometry(1,1);
	let subjects = compute_subjs("", currentlevel);
	currentsubjects = subjects;
	let num = subjects.length;
	// let sizes = compute_sizes(subjects.length);
	// nrows = sizes[0];
	// ncols = sizes[1];
	nrows = compute_size(num);
	ncols = nrows;
	const boundboxes = compute_bbs(nrows,ncols,width,height);
	//console.log(subj2s.length)
	//let list_images = [];
	for (let i = 0; i < nrows; i++) {
		for (let j = 0; j < ncols; j++) {
			let pos = (i*ncols+j);
			if (pos < num) {
				let subject = subjects[pos];
				let list_images = subj1id[subject];
				//console.log(list_images);
				let intersects = getIntKeys(list_images, id2url);
				//console.log(intersects);
				//console.log(pos)
				if (id2url[intersects[0]] != undefined) {
					const first_texture = loader.load(id2url[intersects[0]]);
					//console.log(id2url[intersects[0]]);
					//console.log(first_texture);
					var material = new THREE.MeshBasicMaterial({map:first_texture});
				}
				else {
					//console.log(texture3);
					var material = new THREE.MeshBasicMaterial({map:texture2});
				}
				let cube = new THREE.Mesh( geometry, material );
				level0.add( cube );


				cube.name = "tile-0"+"-"+pos.toString();
				//console.log(cube.name);
				cube.scale.set(width/ncols,height/nrows,0.1);
				cube.position.set((boundboxes[i*ncols+j][0]+boundboxes[i*ncols+j][1])/2,(boundboxes[i*ncols+j][2]+boundboxes[i*ncols+j][3])/2,deg_abstr[subject]*2 - 2);
			}
		}
	}
	scene.add(level0);
	level0.position.z = 0;
//	return livello2;
}

function create_level_one(subj) {
	const geometry = new THREE.PlaneGeometry(1,1);
	let subjects = compute_subjs(subj,1);
	currentsubjects = subjects;
	//console.log(subjects);
	let num = subjects.length;
	// let sizes = compute_sizes(subjects.length);
	// nrows = sizes[0];
	// ncols = sizes[1];
	nrows = compute_size(num);
	ncols = nrows;
	const boundboxes = compute_bbs(nrows,ncols,width,height);
	//console.log(subj2s.length)
	//let list_images = [];
	for (let i = 0; i < nrows; i++) {
		for (let j = 0; j < ncols; j++) {
			let pos = (i*ncols+j);
			if (pos < num) {
				let subject = subjects[pos];
				if (subject in subj2id) {
					console.log(subject);
					let list_images = subj2id[subject];
					//console.log(list_images);
					let intersects = [];
					if (list_images.length > 0){
						console.log(list_images.length);
						intersects = getIntKeys(list_images, id2url);
					} else {
						console.log('no image');
						intersects = ['P04699'];
					}
					if (id2url[intersects[0]] != undefined) {
						const first_texture = loader.load(id2url[intersects[0]]);
						//console.log(id2url[intersects[0]][0]);
						//console.log(first_texture);
						var material = new THREE.MeshBasicMaterial({map:first_texture});
					}
					else {
						//console.log(texture3);
						var material = new THREE.MeshBasicMaterial();
					}
					let cube = new THREE.Mesh( geometry, material );
					level1.add( cube );


					cube.name = "tile-1"+"-"+pos.toString();
					//console.log(cube.name);
					let max_colrow = Math.max(ncols, nrows);
					cube.scale.set(width/max_colrow,height/max_colrow,0.1);
					cube.position.set((boundboxes[i*ncols+j][0]+boundboxes[i*ncols+j][1])/2,(boundboxes[i*ncols+j][2]+boundboxes[i*ncols+j][3])/2,deg_abstr[subject]*3 - 2.5);
				}
			}
		}
	}
	scene.add(level1);
	level1.position.z = 0;
}


function create_level_two(subj,flag) {
	const geometry = new THREE.PlaneGeometry(1,1);
	//console.log(subj);
	let subsubjs = order2subj[subj][sort_category];
	//console.log(subsubjs);
	let num = subsubjs.length;

	// let sizes = compute_sizes(subsubjs.length);
	// nrows = sizes[0];
	// ncols = sizes[1];
	nrows = compute_size(num);
	ncols = nrows;
	const boundboxes = compute_bbs(nrows,ncols,width,height);
	//console.log(subsubjs);
	let intersects = getIntKeys(subsubjs, id2url);
	let l_images = [];
	for (let i = 0; i < nrows; i++) {
		for (let j = 0; j < ncols; j++) {
			let pos = (i*ncols+j);
			if (pos < num) {
				if (id2url[intersects[pos]] != undefined) {
						const first_texture = loader.load(id2url[intersects[pos]]);
						var material = new THREE.MeshBasicMaterial({map:first_texture});
						l_images.push(intersects[pos]);
					}
					else {
						var material = new THREE.MeshBasicMaterial();
					}
					let cube = new THREE.Mesh( geometry, material );
					level2.add( cube );

					cube.name = "tile-2"+"-"+pos.toString();
					//console.log(cube.name);
					let max_colrow = Math.max(ncols, nrows);
					cube.scale.set(width/max_colrow,height/max_colrow,0.1);
					cube.position.set((boundboxes[i*ncols+j][0]+boundboxes[i*ncols+j][1])/2,(boundboxes[i*ncols+j][2]+boundboxes[i*ncols+j][3])/2,0);
				}
			}
		}
	scene.add(level2);
	currentsubjects = l_images;
	level2.position.z = 0;
	if (flag) {
		addButtons(subj);
	}
}

function create_level_three(img) {
	const geometry = new THREE.PlaneGeometry(1,1);
	let image = id2url[img];
	nrows = 2;
	ncols = 1;
	const boundboxes = compute_bbs(nrows,ncols,width,height);
	const first_texture = loader.load(image);
	var material = new THREE.MeshBasicMaterial({map:first_texture});
	let cube = new THREE.Mesh( geometry, material );
	level3.add( cube );
	cube.name = "tile-3"+"-1".toString();
	let max_colrow = Math.max(ncols, nrows);
	cube.scale.set(width/max_colrow,height/max_colrow,0.1);
	cube.position.set((boundboxes[0*ncols][0]+boundboxes[0*ncols][1])/2,(boundboxes[0*ncols][2]+boundboxes[0*ncols][3])/2,0);
	scene.add(level3);
	level3.position.z = 0;
	console.log( '<br /><b>Creation year</b>: '+ metadata[img]['year'].toString());
	let metadata_img = '<b>Title</b>: '+ metadata[img]['title'] + '<br /><b>Artist</b>: '+metadata[img]['artist'] + '<br /><b>Creation year</b>: '+ metadata[img]['year'].toString() + '<br /><b>Acquisition year</b>: '+ metadata[img]['acquisitionYear'].toString()  + '<br /><b>Medium</b>: '+ metadata[img]['medium']  + '<br /><b>Subjects</b>: '+ metadata[img]['subjects'] + '<br /><b>Subjects second level</b>: '+ metadata[img]['subjects_2'];
	middleDiv.hidden = false;
	middleDiv.innerHTML = metadata_img;

}
