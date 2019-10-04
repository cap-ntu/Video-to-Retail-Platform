import React from "react";
import PropTypes from "prop-types";
import TextField from "@material-ui/core/es/TextField/TextField";
import Grid from "@material-ui/core/Grid";
import Avatar from "@material-ui/core/es/Avatar/Avatar";
import withStyles from "@material-ui/core/styles/withStyles";
import Typography from "@material-ui/core/es/Typography/Typography";
import Groups from "./formComponent/Groups";
import CardContent from "@material-ui/core/es/CardContent/CardContent";
import Button from "../../common/Button";
import PermissionControl from "./formComponent/PermissionControl";
import Dialog from "@material-ui/core/Dialog";
import DialogTitle from "@material-ui/core/DialogTitle";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";
import DialogActions from "@material-ui/core/DialogActions";
import Name from "./formComponent/Name";

const styles = theme => ({
    header: {},
    textField: {},
    section: {},
    sectionInner: {},
    dialogPaper: {
        backgroundColor: theme.palette.grey[200],
        opacity: 0.98,
    }
});

class UserEdit extends React.PureComponent {

    state = {
        id: "",
        firstName: "",
        lastName: "",
        username: "",
        email: "",
        is_superuser: false,
        domain: false,
        status: false,
        groups: [],
        validation: {
            name: {error: false, rest: undefined},
            groups: {error: false, rest: undefined},
            permissionControl: {error: false, rest: undefined},
        },
        deleteDialog: false,
        ...this.props.user,
    };

    child = {
        name: React.createRef(),
        permissionControl: React.createRef(),
    };

    handleChange = name => event => {
        this.setState({[name]: event.target.value});
    };

    collectValidation = name => _state => {
        this.setState(state => ({...state, validation: {...state.validation, [name]: _state}}));
    };

    handleDelete = () => {
        this.props.handleDelete(this.state.id);
        this.setState({deleteDialog: false});
    };

    handleSubmit = () => {

        const result = [];

        // call for validate, collect validation
        Object.values(this.child).forEach(v =>
            result.push(v.current.validate()));

        // check errors
        if (result.every(x => !x))
            this.props.handleDone({...this.state, validation: undefined, deleteDialog: undefined})
    };

    render() {
        const {classes, handleCancel} = this.props;
        const {
            firstName, lastName, username, email, accountType, domain, status, groups,
            deleteDialog, validation
        } = this.state;
        const _validation = {};
        Object.keys(validation).forEach(key => _validation[key] = validation[key].rest);

        return (
            <React.Fragment>
                <div className={classes.header}>
                    <Button color={"primary"} onClick={handleCancel}>
                        Cancel
                    </Button>
                    <div style={{flexGrow: 1}}/>
                    <Button color="primary" onClick={this.handleSubmit}
                            disabled={!Object.values(validation).every(x => !x.error)}>
                        Done
                    </Button>
                </div>
                <CardContent>
                    <div className={classes.section}>
                        <Grid direction="row" spacing={8} alignItems="center" container>
                            <Grid item>
                                <Avatar>{username.slice(0, 2).toUpperCase()}</Avatar>
                            </Grid>
                            <Grid item>
                                <Name classes={{textField: classes.textField}}
                                      innerRef={this.child.name}
                                      firstName={firstName}
                                      lastName={lastName}
                                      handleChange={this.handleChange}
                                      error={_validation.name}
                                      handleUpdateValidation={this.collectValidation("name")}
                                />
                            </Grid>
                        </Grid>
                    </div>

                    <div className={classes.section}>
                        <Typography variant="subtitle1" color="textSecondary">Personal Information</Typography>
                        <div className={classes.sectionInner}>
                            <TextField id={"email"}
                                       className={classes.textField}
                                       label={"Email"}
                                       value={email}
                                       onChange={this.handleChange("email")}
                                       margin={"dense"}
                                       fullWidth/>
                        </div>
                    </div>

                    <div className={classes.section}>
                        <Groups classes={{textField: classes.textField}}
                                groups={groups}
                                onChange={(groups, callback) => this.setState({groups: groups}, callback)}
                                handleUpdateValidation={this.collectValidation("groups")}
                        />
                    </div>

                    <div className={classes.section}>
                        <PermissionControl
                            classes={{textField: classes.textField}}
                            innerRef={this.child.permissionControl}
                            accountType={accountType}
                            domain={domain}
                            status={status}
                            error={_validation.permissionControl}
                            handleChange={this.handleChange}
                            handleUpdateValidation={this.collectValidation("permissionControl")}
                        />
                    </div>

                    <div className={classes.section}>
                        <Grid justify="center" container>
                            <Button onClick={() => this.setState({deleteDialog: true})}>
                                <Typography color="error">Delete User</Typography>
                            </Button>
                        </Grid>

                        {/* delete dialog */}
                        <Dialog classes={{paper: classes.dialogPaper}} open={deleteDialog}
                                onClose={() => this.setState({deleteDialog: false})}>
                            <DialogTitle>{`Delete User "${username}"?`}</DialogTitle>
                            <DialogContent>
                                <DialogContentText>
                                    Deleting user {`"${username}"`} will also delete all its data.
                                    This action cannot be undo.
                                </DialogContentText>
                                <DialogActions>
                                    <Button color={"primary"} onClick={() => this.setState({deleteDialog: false})}>
                                        Cancel
                                    </Button>
                                    <Button onClick={this.handleDelete}>
                                        <Typography color={"error"}>Delete Anyway</Typography>
                                    </Button>
                                </DialogActions>
                            </DialogContent>
                        </Dialog>

                    </div>

                </CardContent>
            </React.Fragment>
        );
    }
}

UserEdit.propTypes = {
    classes: PropTypes.object.isRequired,
    user: PropTypes.object.isRequired,
    handleDone: PropTypes.func.isRequired,
    handleCancel: PropTypes.func.isRequired,
    handleDelete: PropTypes.func.isRequired,
};

export default withStyles(styles)(UserEdit);
