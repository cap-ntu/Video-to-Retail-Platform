import React from "react";
import TextField from "@material-ui/core/TextField";
import InputAdornment from "@material-ui/core/InputAdornment";
import IconButton from "@material-ui/core/IconButton";
import VisibilityIcon from "@material-ui/icons/VisibilityRounded";
import VisibilityOffIcon from "@material-ui/icons/VisibilityOffRounded";
import {withStyles} from "@material-ui/styles";

const styles = {
    textField: {},
};

const messages = [
    "Valid password",
    "Password need to contain at least 1 alphabet, and 1 digit",
    "Password need to be at least 8 characters",
    "Password may not contain space",
    "Your password should contains at least 1 alphabet, 1 digit and 8 characters",
];

class Password extends React.Component {
    state = {
        showPassword: false,
    };

    validate = () => {
        const {value, handleUpdateValidation} = this.props;

        let code = 4;
        for (const re of [/^(?![\s\S])/, /[\s]+/, /^.{0,7}$/, /(?=^[^0-9]*$)|(?=^[^a-zA-Z]*$)/]) {
            if (re.test(value))
                break;
            code--;
        }

        handleUpdateValidation({error: Boolean(code), rest: code});

        return code;
    };

    render() {
        const {classes, value, error, onChange} = this.props;
        const {showPassword} = this.state;

        return (<TextField id="password"
                           className={classes.textField}
                           type={showPassword ? "text" : "password"}
                           label="Password"
                           value={value}
                           error={Boolean(error)}
                           onChange={onChange}
                           InputProps={{
                               endAdornment: (
                                   <InputAdornment position="end">
                                       <IconButton
                                           aria-label="Toggle password visibility"
                                           onClick={() => this.setState({showPassword: !showPassword})}
                                       >
                                           {showPassword ? <VisibilityIcon/> : <VisibilityOffIcon/>}
                                       </IconButton>
                                   </InputAdornment>
                               ),
                           }}
                           onBlur={this.validate}
                           required
                           helperText={error === undefined ? messages[4] : messages[error]}
                           margin="dense"/>);
    }
}

export default withStyles(styles)(Password);
