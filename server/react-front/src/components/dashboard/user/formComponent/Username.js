import React from "react";
import TextField from "@material-ui/core/TextField";
import withStyles from "@material-ui/styles/withStyles"

const styles = {
    textField: {}
};

class Username extends React.PureComponent {

    validate = () => {
        const {value, handleUpdateValidation} = this.props;
        const error = value === "";

        handleUpdateValidation({error: error, rest: error});

        return error;
    };

    render() {
        const {classes, value, error, onChange} = this.props;

        return (
            <TextField id="username"
                       className={classes.textField}
                       label="Username"
                       value={value}
                       error={error}
                       onChange={onChange}
                       onBlur={this.validate}
                       helperText={error ? "This field is required" : undefined}
                       required
                       margin="dense"/>
        );
    }
}

export default withStyles(styles)(Username);
