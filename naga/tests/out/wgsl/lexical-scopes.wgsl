fn blockLexicalScope(a: bool) {
    {
        {
        }
    }
}

fn ifLexicalScope(a_1: bool) {
    if a_1 {
    }
}

fn loopLexicalScope(a_2: bool) {
    loop {
    }
}

fn forLexicalScope(a_3: f32) {
    var a_4: i32 = 0i;

    loop {
        let _e3 = a_4;
        if (_e3 < 1i) {
        } else {
            break;
        }
        {
        }
        continuing {
            let _e8 = a_4;
            a_4 = (_e8 + 1i);
        }
    }
}

fn whileLexicalScope(a_5: i32) {
    loop {
        if (a_5 > 2i) {
        } else {
            break;
        }
        {
        }
    }
}

fn switchLexicalScope(a_6: i32) {
    switch a_6 {
        case 0: {
        }
        case 1: {
        }
        default: {
        }
    }
    let test = (a_6 == 2i);
}

