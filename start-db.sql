create database plagdetect;
use plagdetect;

CREATE TABLE article (
	id INT(10) NOT NULL PRIMARY KEY, 
	article MEDIUMBLOB,
);

CREATE TABLE plag (
	id INT(10) NOT NULL PRIMARY KEY, 
	fk_article_id INT(10), -- set foreign key
	fragment BLOB,
	offset INT(10),
	length INT(10) 
);

-- insert into article (id, article) values (0, LOAD_FILE("/dataset/suspicious-document00001.txt"));

