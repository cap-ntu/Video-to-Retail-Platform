import React from "react";
import * as PropTypes from "prop-types";
import Typography from "@material-ui/core/Typography";
import Grid from "@material-ui/core/Grid";
import TextField from "@material-ui/core/TextField";
import MenuItem from "@material-ui/core/MenuItem";
import withStyles from "@material-ui/core/styles/withStyles";

const styles = {
    textField: {}
};

const SelectionTextField = ({classes, menuItems, onChange, ...rest}) => (
    <TextField className={classes.textField} onChange={onChange} {...rest} select margin="dense">
        {menuItems.map(option => (
            <MenuItem key={option} value={option}>
                {option}
            </MenuItem>
        ))}
    </TextField>
);

class PermissionControl extends React.PureComponent {

    validate = () => {
        const {accountType, domain, status, handleUpdateValidation} = this.props;

        const temp = [accountType, domain, status].map(textField => textField === "");
        const error = temp.some(x => x);

        handleUpdateValidation({error: error, rest: temp});

        return error;
    };

    render() {
        const {classes, accountType, domain, error, status, handleChange} = this.props;

        return (
            <React.Fragment>
                <Typography variant="subtitle1" color="textSecondary" gutterBottom>
                    User Permission
                </Typography>

                <Grid container>
                    <Grid xs item>
                        <SelectionTextField
                            id="accountType"
                            label="Account Type"
                            classes={classes}
                            menuItems={["Administrator", "User"]}
                            value={accountType}
                            error={error[0]}
                            onChange={handleChange("accountType")}
                            helperText={error[0] ? "Account type has to be selected" : ""}
                            onBlur={this.validate}
                        />
                    </Grid>

                    <Grid xs item>
                        <Grid xs item>
                            <SelectionTextField
                                id="domain"
                                label="Domain"
                                classes={classes}
                                menuItems={["Staff", "Public"]}
                                value={domain}
                                error={error[1]}
                                onChange={handleChange("domain")}
                                helperText={error[1] ? "Domain has to be selected" : ""}
                                onBlur={this.validate}
                            />
                        </Grid>
                    </Grid>

                    <Grid xs item>
                        <SelectionTextField
                            id="status"
                            label="Status"
                            classes={classes}
                            menuItems={["Activated", "Not Activated"]}
                            value={status}
                            error={error[2]}
                            onChange={handleChange("status")}
                            helperText={error[2] ? "Status has to be selected" : ""}
                            onBlur={this.validate}
                        />
                    </Grid>
                </Grid>
            </React.Fragment>
        );
    }
}


PermissionControl.defaultProps = {error: []};

PermissionControl.propTypes = {
    classes: PropTypes.object.isRequired,
    accountType: PropTypes.oneOf(["", "Administrator", "User"]).isRequired,
    domain: PropTypes.oneOf(["", "Staff", "Public"]).isRequired,
    status: PropTypes.oneOf(["", "Activated", "Not Activated"]).isRequired,
    error: PropTypes.array.isRequired,
    handleChange: PropTypes.func.isRequired,
};

export default withStyles(styles)(PermissionControl);
