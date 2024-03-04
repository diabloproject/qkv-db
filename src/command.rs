use anyhow::Context;
use std::iter::{zip, Peekable, Zip};
use thiserror::Error;

static KEYWORDS: &'static [&'static str] = &[
    // Operations
    "CREATE", "INSERT", "SCAN", // Entities
    "DATABASE", "BUCKET", "QUERIES", "KEYS", "VALUES", // Helpers
    "IF", "NOT", "EXISTS", "WITH", "INTO", "INSIDE", "AND",
];

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unexpected token `{token}`")]
    UnexpectedToken {
        line: usize,
        col: usize,
        token: String,
    },
    #[error("Unexpected end of input")]
    UnexpectedEOS,
    #[error("You must specify bucket to insert data to")]
    NoBucketInInsert,
}

#[derive(Debug, PartialEq)]
pub enum IfExists {
    Fail,
    Skip,
}

#[derive(Debug, PartialEq)]
pub enum ScanTargetBucket {
    Hot,
    All,
    Physical(String),
}

#[derive(Debug, PartialEq)]
pub enum Command {
    CreateDatabase {
        name: String,
        properties: PropertyList,
    },
    CreateBucket {
        database: String,
        name: String,
        properties: PropertyList,
    },
    Insert {
        database: String,
        bucket: String,
        entries: Vec<(Vec<f32>, Vec<f32>)>,
        properties: PropertyList,
    },
    Scan {
        database: String,
        bucket: ScanTargetBucket,
        queries: Vec<Vec<f32>>,
        properties: PropertyList,
    },
    Dummy,
}

#[derive(Debug, PartialEq, Clone)]
pub enum PropertyValue {
    Integer(i32),
    Float(f32),
    String(String),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Property {
    pub name: String,
    pub data: PropertyValue,
}

pub type PropertyList = Vec<Property>;

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
struct AstVecData(Vec<Vec<f32>>);

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
struct AstWithClauseData(PropertyList);

#[derive(Debug, Clone, PartialEq)]
struct AstRefData {
    bucket: Option<String>,
    database: String,
}

#[derive(Debug, Clone)]
enum Token {
    Keyword(String),
    Identifier(String),
    Punctuation(String),
    Number(String),
}

impl Token {
    pub fn ty(&self) -> &'static str {
        match self {
            Token::Keyword(_) => "keyword",
            Token::Identifier(_) => "identifier",
            Token::Punctuation(_) => "punctuation",
            Token::Number(_) => "number",
        }
    }

    pub fn content(&self) -> &str {
        match self {
            Token::Keyword(c) => c.as_str(),
            Token::Identifier(c) => c.as_str(),
            Token::Punctuation(c) => c.as_str(),
            Token::Number(c) => c.as_str(),
        }
    }
}

impl Command {
    fn parse_vec(content: &mut impl Iterator<Item = Token>) -> Result<AstVecData, ParseError> {
        let left_par = content.next();
        if left_par.is_none() {
            return Err(ParseError::UnexpectedEOS);
        }
        let left_par = left_par.unwrap();
        if left_par.ty() != "punctuation" || left_par.content() != "(" {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: 0,
                token: left_par.content().to_string(),
            });
        };

        let mut data: Vec<Vec<f32>> = vec![];

        loop {
            let tok = content.next();
            if tok.is_none() {
                return Err(ParseError::UnexpectedEOS);
            }
            let tok = tok.unwrap();
            if tok.ty() == "punctuation" && tok.content() == ")" {
                break;
            }
            if !data.is_empty() {
                if tok.ty() != "punctuation" || tok.content() != "," {
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        token: tok.content().to_string(),
                    });
                }
                let tok_ = content.next();
                if tok_.is_none() {
                    return Err(ParseError::UnexpectedEOS);
                }
            }

            if tok.ty() != "punctuation" || tok.content() != "[" {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: 0,
                    token: tok.content().to_string(),
                });
            }

            let mut numbers: Vec<f32> = vec![];
            loop {
                let tok = content.next();
                if tok.is_none() {
                    return Err(ParseError::UnexpectedEOS);
                }
                let mut tok = tok.unwrap();
                if tok.content() == "]" {
                    let right_bracket = tok;
                    if right_bracket.ty() != "punctuation" || right_bracket.content() != "]" {
                        return Err(ParseError::UnexpectedToken {
                            line: 0,
                            col: 0,
                            token: right_bracket.content().to_string(),
                        });
                    };
                    break;
                }
                if !numbers.is_empty() {
                    if tok.ty() != "punctuation" || tok.content() != "," {
                        return Err(ParseError::UnexpectedToken {
                            line: 0,
                            col: 0,
                            token: tok.content().to_string(),
                        });
                    }
                    let tok_ = content.next();
                    if tok_.is_none() {
                        return Err(ParseError::UnexpectedEOS);
                    }
                    tok = tok_.unwrap();
                }
                if tok.ty() != "number" {
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        token: tok.content().to_string(),
                    });
                }
                numbers.push(tok.content().parse().expect("Failed to parse float."))
            }
            data.push(numbers);
        }
        Ok(AstVecData(data))
    }

    fn parse_with_clause(
        content: &mut Peekable<impl Iterator<Item = Token>>,
    ) -> Result<AstWithClauseData, ParseError> {
        if let Some(_with) =
            content.next_if(|tok| tok.ty() == "keyword" && tok.content().to_uppercase() == "WITH")
        {
            let name = content.next();
            if name.is_none() {
                return Err(ParseError::UnexpectedEOS);
            }
            let name = name.unwrap();
            if name.ty() != "identifier" {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: 0,
                    token: name.content().to_string(),
                });
            }

            let name = name.content().to_string();

            let eq_sign = content.next();
            if eq_sign.is_none() {
                return Err(ParseError::UnexpectedEOS);
            }
            let eq_sign = eq_sign.unwrap();
            if eq_sign.ty() != "punctuation" || eq_sign.content() != "=" {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: 0,
                    token: eq_sign.content().to_string(),
                });
            }

            let value = content.next();
            if value.is_none() {
                return Err(ParseError::UnexpectedEOS);
            }
            let value = value.unwrap();
            if value.ty() != "number" {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: 0,
                    token: value.content().to_string(),
                });
            }

            let value = value.content().to_string();

            let mut props = vec![Property {
                name,
                data: if value.contains('.') {
                    PropertyValue::Float(value.parse().expect("Failed to parse float value."))
                } else {
                    PropertyValue::Integer(value.parse().expect("Failed to parse integer."))
                },
            }];
            while let Some(_and) = content
                .next_if(|tok| tok.ty() == "keyword" && tok.content().to_uppercase() == "AND")
            {
                let name = content.next();
                if name.is_none() {
                    return Err(ParseError::UnexpectedEOS);
                }
                let name = name.unwrap();
                if name.ty() != "identifier" {
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        token: name.content().to_string(),
                    });
                }

                let name = name.content().to_string();

                let eq_sign = content.next();
                if eq_sign.is_none() {
                    return Err(ParseError::UnexpectedEOS);
                }
                let eq_sign = eq_sign.unwrap();
                if eq_sign.ty() != "punctuation" || eq_sign.content() != "=" {
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        token: eq_sign.content().to_string(),
                    });
                }

                let value = content.next();
                if value.is_none() {
                    return Err(ParseError::UnexpectedEOS);
                }
                let value = value.unwrap();
                if value.ty() != "number" {
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        token: value.content().to_string(),
                    });
                }

                let value = value.content().to_string();
                props.push(Property {
                    name,
                    data: if value.contains('.') {
                        PropertyValue::Float(value.parse().expect("Failed to parse float value."))
                    } else {
                        PropertyValue::Integer(value.parse().expect("Failed to parse integer."))
                    },
                })
            }
            Ok(AstWithClauseData(props))
        } else {
            Ok(AstWithClauseData(vec![]))
        }
    }

    fn parse_ref(
        content: &mut Peekable<impl Iterator<Item = Token>>,
    ) -> Result<AstRefData, ParseError> {
        let tok = content.next();
        if tok.is_none() {
            return Err(ParseError::UnexpectedEOS);
        };
        let tok = tok.unwrap();
        if tok.ty() != "identifier" {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: 0,
                token: tok.content().to_string(),
            });
        };

        let id_1 = tok.content().to_string();

        if let Some(_) = content.next_if(|tok| tok.content().to_uppercase() == "INSIDE") {
            let tok = content.next();
            if tok.is_none() {
                Err(ParseError::UnexpectedEOS)
            } else {
                let tok = tok.unwrap();
                let id_2 = tok.content().to_string();
                Ok(AstRefData {
                    bucket: Some(id_1),
                    database: id_2,
                })
            }
        } else {
            Ok(AstRefData {
                bucket: None,
                database: id_1,
            })
        }
    }

    fn force_keyword(name: Option<&str>, token: Option<Token>) -> Result<String, ParseError> {
        if token.is_none() {
            return Err(ParseError::UnexpectedEOS);
        }
        let token = token.unwrap();
        if token.ty() != "keyword" {
            return Err(ParseError::UnexpectedToken {
                line: 0,
                col: 0,
                token: token.content().to_string(),
            });
        };
        if let Some(name) = name {
            if token.content() != name {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: 0,
                    token: token.content().to_string(),
                });
            }
        };
        Ok(token.content().to_string())
    }

    pub fn parse(content: &str) -> Result<Self, ParseError> {
        let mut tokens = vec![];
        {
            #[derive(Debug, Eq, PartialEq)]
            enum TokenType {
                Keyword,
                Identifier,
                Punctuation,
                Unknown,
                Number,
            }

            let mut line_counter: usize = 0;
            let mut char_counter: usize = 0;

            // Tokenize command
            let mut buff = String::new();
            let mut token_type = TokenType::Unknown;
            for c in content.chars() {
                if token_type != TokenType::Punctuation {
                    if buff.is_empty() && (c.is_alphabetic() || c == '_') {
                        buff.push(c);
                        token_type = TokenType::Identifier;
                    } else if buff.is_empty() && c.is_numeric() || c == '-' {
                        token_type = TokenType::Number;
                        buff.push(c);
                    } else if !buff.is_empty() && c.is_alphanumeric() || c == '_' {
                        buff.push(c);
                    } else if !buff.is_empty() && c.is_numeric() || c == '.' {
                        if c == '.' && buff.contains(c) {
                            return Err(ParseError::UnexpectedToken {
                                line: line_counter,
                                col: char_counter,
                                token: c.to_string(),
                            });
                        }
                        buff.push(c);
                    } else {
                        if KEYWORDS.contains(&buff.to_ascii_uppercase().as_str()) {
                            token_type = TokenType::Keyword;
                        }
                        match token_type {
                            TokenType::Keyword => {
                                tokens.push(Token::Keyword(buff.to_ascii_uppercase()))
                            }
                            TokenType::Identifier => {
                                if !buff.is_empty() {
                                    tokens.push(Token::Identifier(buff.clone()))
                                }
                            }
                            TokenType::Number => tokens.push(Token::Number(buff.clone())),
                            _ => {}
                        }
                        buff.clear();
                        token_type = TokenType::Unknown;
                        if ",.[](){}=;".contains(c) {
                            tokens.push(Token::Punctuation(c.into()))
                        } else if c.is_whitespace() {
                            continue;
                        } else {
                            return Err(ParseError::UnexpectedToken {
                                line: line_counter,
                                col: char_counter,
                                token: c.into(),
                            });
                        }
                    }
                } else {
                }
            }
        }

        // AST
        {
            #[derive(Debug)]
            enum EntityType {
                Bucket,
                Database,
            }
            #[derive(Debug)]
            enum CommandPrototype {
                Create {
                    ref_: AstRefData,
                    with: AstWithClauseData,
                },
                Insert {
                    ref_: AstRefData,
                    keys: AstVecData,
                    values: AstVecData,
                    with: AstWithClauseData,
                },
                Scan {
                    ref_: AstRefData,
                    queries: AstVecData,
                    with: AstWithClauseData,
                },
            }

            let mut token_iter = tokens.into_iter().peekable();
            let tok = token_iter.next();
            if tok.is_none() {
                return Err(ParseError::UnexpectedEOS);
            }
            let tok = tok.unwrap();
            if tok.ty() != "keyword" {
                return Err(ParseError::UnexpectedToken {
                    line: 0,
                    col: 0,
                    token: tok.content().to_string(),
                });
            }
            let command_prototype = match tok.content().to_uppercase().as_str() {
                "CREATE" => {
                    let entity = Command::force_keyword(None, token_iter.next())?;
                    let ref_ = match entity.to_uppercase().as_str() {
                        "DATABASE" => {
                            let ref_ = Command::parse_ref(&mut token_iter)?;
                            if ref_.bucket.is_some() {
                                return Err(ParseError::UnexpectedToken {
                                    line: 0,
                                    col: 0,
                                    token: "INSIDE".to_string(),
                                });
                            }
                            ref_
                        }
                        "BUCKET" => {
                            let ref_ = Command::parse_ref(&mut token_iter)?;
                            if ref_.bucket.is_none() {
                                let tok = token_iter.next();
                                return if let Some(tok) = tok {
                                    Err(ParseError::UnexpectedToken {
                                        line: 0,
                                        col: 0,
                                        token: tok.content().to_string(),
                                    })
                                } else {
                                    Err(ParseError::UnexpectedEOS)
                                };
                            }
                            ref_
                        }
                        tok => {
                            return Err(ParseError::UnexpectedToken {
                                line: 0,
                                col: 0,
                                token: tok.to_string(),
                            })
                        }
                    };
                    let with = Command::parse_with_clause(&mut token_iter)?;
                    CommandPrototype::Create { ref_, with }
                }
                "INSERT" => {
                    let into = token_iter.next();
                    if into.is_none() {
                        return Err(ParseError::UnexpectedEOS);
                    }
                    let into = into.unwrap();
                    if into.content().to_uppercase() != "INTO" {
                        return Err(ParseError::UnexpectedToken {
                            line: 0,
                            col: 0,
                            token: into.content().to_string(),
                        });
                    }
                    let ref_ = Command::parse_ref(&mut token_iter)?;
                    Command::force_keyword(Some("KEYS"), token_iter.next())?;
                    let keys = Command::parse_vec(&mut token_iter)?;
                    Command::force_keyword(Some("VALUES"), token_iter.next())?;
                    let values = Command::parse_vec(&mut token_iter)?;
                    let with = Command::parse_with_clause(&mut token_iter)?;
                    CommandPrototype::Insert {
                        ref_,
                        keys,
                        values,
                        with,
                    }
                }
                "SCAN" => {
                    let ref_ = Command::parse_ref(&mut token_iter)?;
                    Command::force_keyword(Some("QUERIES"), token_iter.next())?;
                    let queries = Command::parse_vec(&mut token_iter)?;
                    let with = Command::parse_with_clause(&mut token_iter)?;
                    CommandPrototype::Scan {
                        ref_,
                        queries,
                        with,
                    }
                }
                x => {
                    return Err(ParseError::UnexpectedToken {
                        line: 0,
                        col: 0,
                        token: x.into(),
                    })
                }
            };
            return Ok(match command_prototype {
                CommandPrototype::Create { ref_, with } => match ref_.bucket {
                    None => Command::CreateDatabase {
                        name: ref_.database,
                        properties: with.0,
                    },

                    Some(bucket) => Command::CreateBucket {
                        database: ref_.database,
                        name: bucket,
                        properties: with.0,
                    },
                },

                CommandPrototype::Insert {
                    ref_,
                    keys,
                    values,
                    with,
                } => {
                    if ref_.bucket.is_none() {
                        return Err(ParseError::NoBucketInInsert);
                    }
                    Command::Insert {
                        database: ref_.database,
                        bucket: ref_.bucket.unwrap(),
                        entries: zip(keys.0.into_iter(), values.0.into_iter()).collect::<Vec<(
                            Vec<f32>,
                            Vec<f32>,
                        )>>(
                        ),
                        properties: with.0,
                    }
                }
                CommandPrototype::Scan {
                    ref_,
                    queries,
                    with,
                } => Command::Scan {
                    database: ref_.database,
                    bucket: {
                        if let Some(bucket) = ref_.bucket {
                            match bucket.as_str() {
                                "ALL" => ScanTargetBucket::All,
                                "HOT" => ScanTargetBucket::Hot,
                                b => ScanTargetBucket::Physical(b.to_string()),
                            }
                        } else {
                            ScanTargetBucket::All
                        }
                    },
                    queries: queries.0,
                    properties: with.0,
                },
            });
        }
    }
}

pub fn parse_commands(content: &str) -> Result<Vec<Command>, ParseError> {
    let mut commands = vec![];
    let mut prev = String::new();
    let mut is_comment = false;
    for c in content.chars() {
        if c == '/' && prev.ends_with('/') {
            // Comment starts with "//"
            is_comment = true;
            prev.remove(prev.len() - 1);
        } else if c == '\n' && is_comment {
            // Comment ends by new line
            is_comment = false;
        } else if c == ';' && !is_comment {
            // End if command
            prev.push(c);
            commands.push(Command::parse(&prev)?);
            prev.clear()
        } else if !is_comment {
            prev.push(c)
        }
    }

    Ok(commands)
}
