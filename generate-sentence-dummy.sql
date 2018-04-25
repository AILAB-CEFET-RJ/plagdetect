drop table sentence;


create table sentence
(
  id            INTEGER not null
    primary key
  autoincrement,
  fk_article_id INT     not null
    references article,
  fragment      TEXT    not null,
  offset        INT     not null,
  length        INT     not null,
  isplag        BOOL    not null
);

INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 1 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 2 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 3 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 4 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 5 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 6 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 7 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 8 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (1, 'Article 1 - Sentence 9 - Plag=TRUE',   0, 1, 1);

INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 1 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 2 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 3 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 4 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 5 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 6 - Plag=FALSE',  0, 1, 0);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 7 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 8 - Plag=TRUE',   0, 1, 1);
INSERT INTO "sentence" ("fk_article_id", "fragment", "offset", "length", "isplag") VALUES (2, 'Article 2 - Sentence 9 - Plag=FALSE',  0, 1, 0);