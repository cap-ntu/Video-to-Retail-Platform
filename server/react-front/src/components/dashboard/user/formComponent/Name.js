import React from "react";
import TextField from "@material-ui/core/TextField";
import {withStyles} from "@material-ui/styles";

const styles = {
    textField: {}
};

class Name extends React.PureComponent {

    validate = () => {
        const {firstName, lastName, handleUpdateValidation} = this.props;

        const temp = [firstName, lastName].map(textField => textField === "");
        const error = temp.some(x => x);

        handleUpdateValidation({error: error, rest: temp});

        return error;
    };

    render() {
        const {classes, firstName, lastName, error, handleChange} = this.props;

        return (
            <React.Fragment>
                <TextField id="firstName"
                           className={classes.textField}
                           label="First Name"
                           value={firstName}
                           error={error[0]}
                           onChange={handleChange("firstName")}
                           onBlur={this.validate}
                           required
                           helperText={error[0] ? "This field is required" : undefined}
                           margin="dense"/>
                <TextField id="lastName"
                           className={classes.textField}
                           label="Last Name"
                           value={lastName}
                           error={error[1]}
                           onChange={handleChange("lastName")}
                           onBlur={this.validate}
                           required
                           helperText={error[0] ? "This field is required" : undefined}
                           margin="dense"/>
            </React.Fragment>
        );
    }
}

Name.defaultProps = {error: []};

export default withStyles(styles)(Name);
